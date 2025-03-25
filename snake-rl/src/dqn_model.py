import numpy as np
import random
import tensorflow as tf
from tensorflow.keras import layers
from collections import deque

from game_setup import *
from action_state_spaces import *

# Initialize replay buffer
experience_replay = deque(maxlen=2000)

# DQN Hyperparameters
gamma = 0.95
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01
batch_size = 32
learning_rate = 0.001
num_episodes = 500

# Performance tracking
episode_scores = []

# Define your DQN model
model = tf.keras.Sequential([
    layers.Input(shape=(8,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(4)  # 4 actions
])
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss='mse')

for episode in range(num_episodes):
    state, snake_body, food_position = reset_game()
    state = get_state(snake_body, food_position, width, height, snake_size)
    state = np.reshape(state, [1, 8])
    done = False
    score = 0

    while not done:
        if np.random.rand() < epsilon:
            action = random.randint(0, 3)
        else:
            q_values = model.predict(state, verbose=0)
            action = np.argmax(q_values[0])

        new_state, reward, done, new_food_position = step(action, snake_body, width, height, food_position, snake_size)
        
        print(f"new_state, reward, done, new_food_position = {new_state, reward, done, new_food_position}")
        food_position = new_food_position
        new_state = np.reshape(new_state, [1, 8])

        experience_replay.append((state, action, reward, new_state, done))
        state = new_state
        score += reward

        if len(experience_replay) > batch_size:
            minibatch = random.sample(experience_replay, batch_size)
            states = np.vstack([x[0] for x in minibatch])
            next_states = np.vstack([x[3] for x in minibatch])

            q_targets = model.predict(states, verbose=0)
            q_next = model.predict(next_states, verbose=0)

            for i, (_, action, reward, _, done_flag) in enumerate(minibatch):
                target = reward
                if not done_flag:
                    target += gamma * np.max(q_next[i])
                q_targets[i][action] = target

            model.fit(states, q_targets, epochs=1, verbose=0)

    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    episode_scores.append(score)

    # --- PRINTS AQUI ---
    print(f"Episode {episode+1}/{num_episodes} | Score: {round(score, 2)} | Epsilon: {round(epsilon, 4)}")

    if (episode + 1) % 10 == 0:
        avg_last_10 = np.mean(episode_scores[-10:])
        print(f"→ Média dos últimos 10 episódios: {round(avg_last_10, 2)}")

# --- AVALIAÇÃO ---
def test_agent(model, num_episodes=10):
    scores = []
    for episode in range(num_episodes):
        state = reset_game()
        score = 0
        done = False

        while not done:
            q_values = model.predict(state, verbose=0)
            action = np.argmax(q_values[0])
            state, reward, done = step(action, snake_body, width, height, food_position, snake_size)
            state = np.reshape(state, [1, 8])
            score += reward

        print(f"[TEST] Episódio {episode+1}: Score = {round(score, 2)}")
        scores.append(score)

    return np.mean(scores)

print("\n===== Avaliação final =====")
scores = test_agent(model=model, num_episodes=10)
print(f"Score médio no teste: {round(scores, 2)}")
