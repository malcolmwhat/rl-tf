# -*- coding: utf-8 -*-

import gym
import numpy as np


def learn_frozen_lake(learning_rate, discount, episodes):
    """Learn the frozen lake using tabular Q-Learning."""
    env = gym.make('FrozenLake-v0')

    # Create Q table based on environment.
    Q = np.zeros([env.observation_space.n, env.action_space.n])
    rewards = []

    for i in range(episodes):
        noise_factor = 1 / (1 + i)  # Noise becomes less large as we progress

        # Reset environment, and observe first state.
        state = env.reset()

        total_reward = 0
        terminated = False
        step = 0

        # Prevent infinite loop I guess...
        while step < 99:
            step += 1

            # Apply noise to action values from current state.
            possible_action_values = Q[state, :]
            noise_vector = np.random.randn(1, env.action_space.n)
            scaled_noise = noise_vector * noise_factor
            noisy_action_values = possible_action_values + scaled_noise

            # Make noisy, greedy choice based on Q table.
            action = np.argmax(noisy_action_values)

            # Attempt to take action, and observe new states and reward.
            new_state, reward, terminated, _ = env.step(action)

            # Bellman update step for the Q values.
            optimal_action_value = np.max(Q[new_state, :])
            discount_action_val = discount * optimal_action_value
            state_value_diff = reward + discount_action_val - Q[state, action]
            Q[state, action] += learning_rate * state_value_diff

            # Update this episode's rewards.
            total_reward += reward
            state = new_state

            if terminated:
                break

        rewards.append(total_reward)

    print("Score over time: " + str(sum(rewards) / episodes))
    print("Final Q-Table Values")
    print(Q)


if __name__ == "__main__":
    learn_frozen_lake(0.8, 0.95, 2000)
