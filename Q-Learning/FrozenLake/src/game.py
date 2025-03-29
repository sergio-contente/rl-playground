import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time

import os

def print_env(env):
    """Print the environment layout for debugging"""
    desc = env.unwrapped.desc.astype(str)
    print("Environment Layout:")
    print("S=Start, F=Frozen (safe), H=Hole, G=Goal")
    for row in desc:
        print(''.join(row))

def run(episodes, is_training=True, render=False):
    # Create environment - using 4x4 as specified in the code
    env = gym.make('FrozenLake-v1', map_name='8x8', is_slippery=False, render_mode='human' if render else None)
    
    # Print the environment layout to understand where holes and goals are
    print_env(env)
    
    # Set maximum steps per episode to prevent infinite loops
    max_steps_per_episode = 100
    
    if is_training:
        q = np.zeros((env.observation_space.n, env.action_space.n))  # init Q-table
    else:
        try:
            with open("frozen_lake8x8.pkl", "rb") as f:
                q = pickle.load(f)
        except FileNotFoundError:
            print("Model file not found. Starting with a new Q-table.")
            q = np.zeros((env.observation_space.n, env.action_space.n))

    # Hyperparameters - adjusted for better performance
    epsilon = 1.0  # Starting exploration rate
    min_epsilon = 0.01  # Lower minimum exploration for more exploration
    epsilon_decay_rate = 0.9995  # Slower decay rate
    
    learning_rate = 0.8  # Higher learning rate
    gamma = 0.95  # Slightly reduced discount factor

    rewards_per_episode = np.zeros(episodes)
    success_count = 0
    steps_per_episode = []

    for i in range(episodes):
        state = env.reset()[0]  # states 0 to 15 for 4x4 grid
        terminated = False
        truncated = False
        
        step_count = 0
        episode_reward = 0
        
        # Episode loop
        while not terminated and not truncated and step_count < max_steps_per_episode:
            step_count += 1
            
            # Exploration-exploitation balance
            if is_training and np.random.random() < epsilon:
                action = env.action_space.sample()  # Explore: random action
            else:
                # Exploit: choose best action (with random tiebreaking to avoid left bias)
                max_value = np.max(q[state, :])
                max_indices = np.where(q[state, :] == max_value)[0]
                action = np.random.choice(max_indices)

            new_state, reward, terminated, truncated, _ = env.step(action)
            
            # Add a small negative reward for each step to encourage shorter paths
            step_reward = -0.01
            
            # Huge reward for reaching the goal
            if reward > 0:  
                step_reward = 1.0
            
            # Additional penalty for falling in a hole
            if terminated and reward == 0:
                step_reward = -1.0
                
            if is_training:
                # Q-learning update with the adjusted reward
                q[state, action] = (1 - learning_rate) * q[state, action] + learning_rate * (
                    step_reward + gamma * np.max(q[new_state, :])
                )

            episode_reward += reward
            state = new_state

            # Track success
            if reward > 0:
                success_count += 1
                rewards_per_episode[i] = 1
                print(f"🎯 GOT REWARD! Episode {i+1}, steps={step_count}, epsilon={epsilon:.4f}")
                
        steps_per_episode.append(step_count)
        
        # Decay epsilon after each episode
        epsilon = max(min_epsilon, epsilon * epsilon_decay_rate)

        # Periodically report progress
        if (i+1) % 100 == 0 or i == 0:
            win_rate = np.sum(rewards_per_episode[max(0, i-99):(i+1)]) / min(100, i+1)
            avg_steps = np.mean(steps_per_episode[-100:]) if steps_per_episode else 0
            print(f"Episode {i+1}: Epsilon = {epsilon:.4f}, Win rate = {win_rate:.2f}, Avg steps = {avg_steps:.1f}")
            
            # Debug Q-values periodically
            if (i+1) % 1000 == 0:
                print("\nCurrent Q-table (sample):")
                for s in range(min(5, env.observation_space.n)):
                    print(f"State {s}: {q[s, :]}")

    env.close()

    # Plot results
    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards_per_episode[max(0, t-100):(t+1)])

     
    plt.figure(figsize=(10, 5))
    plt.plot(sum_rewards)
    plt.title('Rewards over time (100-episode moving window)')
    plt.xlabel('Episodes')
    plt.ylabel('Total rewards')
    plt.grid(True)
    plt.savefig('frozen_lake8x8.png')
    plt.close()

    if is_training:
        with open("frozen_lake8x8.pkl", "wb") as f:
            pickle.dump(q, f)
        
        # print("\nFinal Q-table:")
        # print(np.round(q, 2))
        
        # print(f"\nTotal successes: {success_count} out of {episodes} episodes ({success_count/episodes*100:.2f}%)")
        
        # # Visualize the learned policy
        # print("\nLearned policy:")
        # policy = np.argmax(q, axis=1)
        # actions = ["←", "↓", "→", "↑"]
        # policy_grid = np.array([actions[a] for a in policy]).reshape(8, 8)
        
        # # Print the grid with nicer formatting
        # hole_positions = []
        # goal_position = None
        
        # # Get hole and goal positions from environment
        # desc = env.unwrapped.desc.astype(str)
        # for i, row in enumerate(desc):
        #     for j, cell in enumerate(row):
        #         if cell == 'H':
        #             hole_positions.append(i * 4 + j)
        #         elif cell == 'G':
        #             goal_position = i * 4 + j
        
        # # Print the policy with environment information
        # print("\nPolicy Map (H=Hole, G=Goal, S=Start):")
        # for i in range(4):
        #     row_str = ""
        #     for j in range(4):
        #         pos = i * 4 + j
        #         if pos == 0:
        #             row_str += " S  "
        #         elif pos in hole_positions:
        #             row_str += " H  "
        #         elif pos == goal_position:
        #             row_str += " G  "
        #         else:
        #             row_str += f" {policy_grid[i][j]}  "
        #     print(row_str)

if __name__ == '__main__':
    # Let's try more episodes with improved parameters
    run(10000, is_training=True, render=False)
    
    # Uncomment to test the learned policy
    print("\nTesting learned policy...")
    time.sleep(1)
    run(5, is_training=False, render=True)
