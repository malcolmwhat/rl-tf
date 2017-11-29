# -*- coding: utf-8 -*-

import gym
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt


if __name__ == "__main__":
    env = gym.make("FrozenLake-v0")

    tf.reset_default_graph()

    # Feed forward network
    # Feed in state-> get out a 4x1 vector which are the action values.
    state_input = tf.placeholder(shape=[1, 16], dtype=tf.float32)
    W = tf.Variable(tf.random_uniform([16, 4], 0, 0.01))
    predicted_Q_values = tf.matmul(state_input, W)
    selected_action = tf.argmax(predicted_Q_values, 1)

    # Loss calculation and optimization step
    observed_Q_values = tf.placeholder(shape=[1, 4], dtype=tf.float32)
    loss = tf.reduce_sum(tf.square(observed_Q_values - predicted_Q_values))
    trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    update_model = trainer.minimize(loss)

    init = tf.initialize_all_variables()

    # Learning params
    y = 0.99
    e = 0.1
    num_episodes = 2000

    # Lists for total rewards and steps per episode.
    steps_per_episode = []
    rewards_per_episode = []

    with tf.Session() as sess:
        sess.run(init)  # Initialize the neural network

        for i in range(num_episodes):
            state = env.reset()
            total_reward = 0
            terminated = False
            j = 0

            # The actual walk through the episode.
            while j < 99:
                j += 1
                action, allQ = sess.run([selected_action, predicted_Q_values],
                        feed_dict={state_input: np.identity(16)[state:state+1]})
                if np.random.rand(1) < e:
                    action[0] = env.action_space.sample()

                new_state, reward, terminated, _ = env.step(action[0])

                next_stateQ = sess.run(predicted_Q_values, feed_dict={
                    state_input:np.identity(16)[new_state:new_state+1]})
                max_state_action_value = np.max(next_stateQ)
                target_state_values = allQ

                target_state_values[0, action[0]] = reward + y * max_state_action_value

                _, new_weights = sess.run([update_model, W], feed_dict={
                    state_input:np.identity(16)[state : state + 1],
                    observed_Q_values: target_state_values})
                total_reward += reward

                state = new_state

                if terminated:
                    e = 1 / ((i / 50) + 10)
                    break

            steps_per_episode.append(j)
            rewards_per_episode.append(total_reward)

    print("Percent of succesful episode: {}%".format(
        str(sum(rewards_per_episode)/num_episodes)))

    plt.plot(rewards_per_episode)
    plt.plot(steps_per_episode)
