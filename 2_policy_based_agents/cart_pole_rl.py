# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import gym


def discount_rewards(r, gamma):
    """Calculate the discounted reward of a buffer of reward values."""
    discounted_r = np.zeros_like(r)
    running_sum = 0
    for i in range(r.size - 1, -1, -1):
        running_sum = running_sum * gamma + r[i]
        discounted_r[i] = running_sum

    return discounted_r


class Agent(object):
    def __init__(self, lr, s_size, a_size, h_size):
        tf.reset_default_graph()

        # Feed-forward portion.
        self.state_in = tf.placeholder(shape=[None, s_size], dtype=tf.float32)
        hidden = slim.fully_connected(self.state_in, h_size,
                                      biases_initializer=None,
                                      activation_fn=tf.nn.relu)

        self.output = slim.fully_connected(hidden, a_size,
                                           activation_fn=tf.nn.softmax,
                                           biases_initializer=None)

        self.chosen_action = tf.argmax(self.output, 1)

        # Training procedure.
        self.reward_container = tf.placeholder(shape=[None], dtype=tf.float32)
        self.action_holder = tf.placeholder(shape=[None], dtype=tf.int32)
        self.indices = tf.range(0, tf.shape(self.output)[0]) * tf.shape(
            self.output)[1] + self.action_holder

        self.responsible_outputs = tf.gather(tf.reshape(self.output, [-1]),
                                             self.indices)

        self.loss = -tf.reduce_mean(tf.log(self.responsible_outputs) *
                                    self.reward_container)

        tvars = tf.trainable_variables()

        self.gradient_holders = []
        for idx, var in enumerate(tvars):
            placeholder = tf.placeholder(tf.float32, name=str(idx)+'_holder')
            self.gradient_holders.append(placeholder)

        self.gradients = tf.gradients(self.loss, tvars)

        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        self.update_batch = optimizer.apply_gradients(zip(
            self.gradient_holders, tvars))

        init = tf.global_variables_initializer()
        self.session = tf.Session()
        self.session.run(init)
        self.gradient_buffer = self.session.run(tf.trainable_variables())

        for ix, grad in enumerate(self.gradient_buffer):
            self.gradient_buffer[ix] = grad * 0

    def pick_action(self, s):
        a_dist = self.session.run(self.output, feed_dict={self.state_in:[s]})
        a = np.random.choice(a_dist[0], p=a_dist[0])
        a = np.argmax(a_dist == a)
        return a


def main():
    env = gym.make("CartPole-v0")
    gamma = 0.99

    agent = Agent(lr=1e-2, s_size=4, a_size=2, h_size=8)

    total_episodes = 5000
    max_ep = 999
    update_frequency = 5

    i = 0
    total_reward = []
    total_length = []

    while i < total_episodes:
        s = env.reset()
        running_reward = 0
        episode_history = []

        for j in range(max_ep):
            a = agent.pick_action(s)

            new_state, r, terminate, _ = env.step(a)
            episode_history.append([s, a, r, new_state])
            s = new_state
            running_reward += r

            if terminate:
                # Update the network
                episode_history = np.array(episode_history)
                episode_history[:,2] = discount_rewards(episode_history[:,2],
                                                        gamma)

                feed_in = {agent.reward_container:episode_history[:,2],
                           agent.action_holder:episode_history[:,1],
                           agent.state_in:np.vstack(episode_history[:,0])}

                gradients = agent.session.run(agent.gradients,
                                              feed_dict=feed_in)

                for index, gradient in enumerate(gradients):
                    agent.gradient_buffer[index] += gradient

                if i % update_frequency == 0 and i != 0:
                    feed_in = dict(zip(agent.gradient_holders,
                                       agent.gradient_buffer))

                    _ = agent.session.run(agent.update_batch,
                                          feed_dict=feed_in)

                    for index, gradient in enumerate(agent.gradient_buffer):
                        agent.gradient_buffer[index] = gradient * 0

                total_reward.append(running_reward)
                total_length.append(j)
                break

        if i % 100 == 0:
            print("Reward {}".format(np.mean(total_reward[-100:])))
            print("Avg Len {}".format(np.mean(total_length[-100:])))
        i += 1


if __name__ == "__main__":
    main()
