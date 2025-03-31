import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time

def discretize(value, bins):
    return np.digitize(value, bins) - 1

def run(episodes, is_training=True, render=False):
    env = gym.make('MountainCar-v0', render_mode='human' if render else None)
    
    # Discretize the continuous state space
    n_bins = 20  # Reduced bin count for better generalization
    pos_bins = np.linspace(env.observation_space.low[0], env.observation_space.high[0], n_bins + 1)[1:-1]
    vel_bins = np.linspace(env.observation_space.low[1], env.observation_space.high[1], n_bins + 1)[1:-1]
    
    # Maximum steps per episode
    max_steps_per_episode = 200  # Default for MountainCar-v0
    
    if is_training:
        q = np.zeros((n_bins, n_bins, env.action_space.n))  # Initialize Q-table
    else:
        try:
            with open("mountaincarv0.pkl", "rb") as f:
                q = pickle.load(f)
            print("Loaded Q-table from file.")
        except FileNotFoundError:
            print("Model file not found. Starting with a new Q-table.")
            q = np.zeros((n_bins, n_bins, env.action_space.n))

    # Hyperparameters
    learning_rate = 0.1
    gamma = 0.99          # Higher discount factor for long-term rewards
    epsilon = 1.0         # Initial exploration rate
    epsilon_min = 0.01
    epsilon_decay = 0.9995
    
    rewards_per_episode = np.zeros(episodes)
    success_count = 0
    steps_per_episode = []

    for i in range(episodes):
        state, _ = env.reset()
        state_p = discretize(state[0], pos_bins)
        state_v = discretize(state[1], vel_bins)
        
        terminated = False
        truncated = False
        step_count = 0
        total_reward = 0
        
        # Run episode
        while not (terminated or truncated) and step_count < max_steps_per_episode:
            step_count += 1
            
            # Explore or exploit
            if is_training and np.random.random() < epsilon:
                action = env.action_space.sample()  # Explore
            else:
                action = np.argmax(q[state_p, state_v])  # Exploit
            
            # Take action
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state_p = discretize(next_state[0], pos_bins)
            next_state_v = discretize(next_state[1], vel_bins)
            
            # Custom reward structure to guide learning
            position = next_state[0]
            velocity = next_state[1]
            height = np.sin(3 * position) * 0.45 + 0.55  # Convert position to height
            
            # Dense reward based on:
            # 1. Height gained (position closer to goal)
            # 2. Velocity in the correct direction (positive velocity if on right side, negative if on left)
            modified_reward = reward  # Base environment reward
            
            # Height reward
            if position < -0.5:  # Left side of the valley
                modified_reward += 0.1 * (height + 0.5)  # Reward for climbing the left hill
                direction_bonus = max(0, -velocity) * 0.1  # Reward for moving left when on left side
            else:  # Right side (toward the goal)
                modified_reward += 0.1 * (height + position)  # Reward for climbing toward the goal
                direction_bonus = max(0, velocity) * 0.1  # Reward for moving right when on right side
            
            modified_reward += direction_bonus + abs(velocity) * 0.1  # Reward for building momentum
            
            if terminated and position >= 0.5:  # Reached the goal
                modified_reward += 10.0
                success_count += 1
                rewards_per_episode[i] = 1
                print(f"🎯 Success! Episode {i+1}, steps={step_count}, epsilon={epsilon:.4f}")
            
            # Q-learning update
            if is_training:
                best_next_action = np.argmax(q[next_state_p, next_state_v])
                td_target = modified_reward + gamma * q[next_state_p, next_state_v, best_next_action]
                td_error = td_target - q[state_p, state_v, action]
                q[state_p, state_v, action] += learning_rate * td_error
            
            total_reward += reward
            state = next_state
            state_p = next_state_p
            state_v = next_state_v
        
        # Decay epsilon
        if is_training:
            epsilon = max(epsilon_min, epsilon * epsilon_decay)
        
        steps_per_episode.append(step_count)
        
        # Periodically report progress
        if (i+1) % 100 == 0 or i == 0:
            win_rate = np.sum(rewards_per_episode[max(0, i-99):(i+1)]) / min(100, i+1)
            avg_steps = np.mean(steps_per_episode[-100:]) if steps_per_episode else 0
            print(f"Episode {i+1}: Epsilon = {epsilon:.4f}, Win rate = {win_rate:.2f}, Avg steps = {avg_steps:.1f}")
            
            # Show sample Q-values every 1000 episodes
            if (i+1) % 1000 == 0:
                print("\nSample Q-values:")
                mid_pos = n_bins // 2
                
                # Show Q-values for different velocity states at middle position
                for vel_idx in range(0, n_bins, n_bins//5):
                    q_vals = q[mid_pos, vel_idx]
                    vel_val = vel_bins[vel_idx] if vel_idx < len(vel_bins) else "high"
                    print(f"Position: middle, Velocity: {vel_val:.2f} → Actions: {np.round(q_vals, 2)}")
                
                # Show Q-values for different positions at zero velocity
                zero_vel = n_bins // 2
                for pos_idx in range(0, n_bins, n_bins//5):
                    q_vals = q[pos_idx, zero_vel]
                    pos_val = pos_bins[pos_idx] if pos_idx < len(pos_bins) else "right"
                    print(f"Position: {pos_val:.2f}, Velocity: ~0 → Actions: {np.round(q_vals, 2)}")

    env.close()

    # Plot results
    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards_per_episode[max(0, t-99):(t+1)]) / min(100, t+1)
    
    plt.figure(figsize=(10, 5))
    plt.plot(sum_rewards)
    plt.title('Success Rate (100-episode moving average)')
    plt.xlabel('Episodes')
    plt.ylabel('Success Rate')
    plt.grid(True)
    plt.savefig('mountaincar_performance.png')
    plt.close()

    # Save the Q-table
    if is_training:
        with open("mountaincarv0.pkl", "wb") as f:
            pickle.dump(q, f)
        print("Model saved.")

if __name__ == '__main__':
    # Train the agent
    run(5000, is_training=True, render=False)
    
    # Test the learned policy
    print("\nTesting learned policy...")
    time.sleep(1)
    run(5, is_training=False, render=True)
