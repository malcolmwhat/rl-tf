# -*- coding: UTF-8 -*-

"""Contextual Bandit Problem.

This module is based on tutorial 1.5 of Reinforcement Learning with TF.

The point is to lead into the full RL problem from the bandit problem.
While in the previous tutorial (found in learn_bandits.py) we covered
systems where actions mapped directly to rewards, the point here is to
see what happens when the state and action both impact the reward of
the problem. The key difference between this and the full RL problem is
in this case the action has no impact on the state of the environment.

In this formulation, there are multiple bandits (states) and we know
exactly which bandit we are dealing with while choosing actions.
"""


import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np


class ContextualBandit(object):
    def __init__(self, bandit_list):
        self.state = 0
        self.bandits = np.array(bandit_list)

        self.num_bandits = self.bandits.shape[0]
        self.num_actions = self.bandits.shape[1]

    def update_state(self):
        """Update the state of the problem, i.e. select a bandit at random."""
        self.state = np.random.randint(0, self.num_bandits)
        return self.state

    def pull_arm(self, action):
        """Pull the <action>th arm of the current bandit."""
        bandit_action_value = self.bandits[self.state, action]
        result = np.random.randn(1)

        reward = 1 if result > bandit_action_value else -1
        return reward


class Agent(object):
    """An agent to play the contextual bandit game."""
    def __init__(self, lr, s_size, a_size, banditos):
        tf.reset_default_graph()
        self.bandits = banditos
        self.state_in = tf.placeholder(shape=[1], dtype=tf.int32)
        self.state_in_one_hot = slim.one_hot_encoding(self.state_in, s_size)
        output = slim.fully_connected(self.state_in_one_hot, a_size,
                                      biases_initializer=None,
                                      activation_fn=tf.nn.sigmoid,
                                      weights_initializer=tf.ones_initializer())

        self.output = tf.reshape(output, [-1])
        self.chosen_action = tf.argmax(self.output, 0)

        # The next six lines establish the training proceedure.
        # We feed the reward and chosen action into the network
        # to compute the loss, and use it to update the network.
        self.reward_holder = tf.placeholder(shape=[1], dtype=tf.float32)
        self.action_holder = tf.placeholder(shape=[1], dtype=tf.int32)
        self.responsible_weight = tf.slice(self.output, self.action_holder, [1])
        self.loss = -(tf.log(self.responsible_weight) * self.reward_holder)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
        self.update = optimizer.minimize(self.loss)
        self.weights = None
        self.session = None
        self.optimal_weight = None
        self.weights = tf.trainable_variables()[0]
        self.init = tf.global_variables_initializer()

    def init_tf_graph(self):
        self.session = tf.Session()
        self.session.run(self.init)

    def pick_action(self, s, e):
        """Pick an action based on the current state and random selection factor."""
        if np.random.rand(1) < e:
            return np.random.randint(self.bandits.num_actions)
        else:
            return self.session.run(self.chosen_action, feed_dict={self.state_in: [s]})

    def train_network(self, state, action, reward):
        input_dict = {self.reward_holder: [reward],
                      self.action_holder: [action],
                      self.state_in: [state]}
        _, ww = self.session.run([self.update, self.weights], feed_dict=input_dict)
        return ww


def print_mean_rewards(bandits, rewards):
    print("Mean reward for each of the {} bandits: {}".format(
        str(bandits.num_bandits), str(np.mean(rewards, axis=1))
    ))


if __name__ == '__main__':
    bandits = ContextualBandit([[0.2, 0.0, -0.0, -5.0], [0.1, -5, 1, 0.25],
                               [-5, 5, 5, 5]])
    agent = Agent(lr=0.001, s_size=bandits.num_bandits, a_size=bandits.num_actions, banditos=bandits)

    episodes = 10000

    # Scores for bandits.
    total_rewards = np.zeros([bandits.num_bandits, bandits.num_actions])
    e = 0.1

    agent.init_tf_graph()

    for i in range(episodes):
        state = bandits.update_state()

        action = agent.pick_action(state, e)
        reward = bandits.pull_arm(action)
        ww = agent.train_network(state, action, reward)

        total_rewards[state, action] += reward

        if i % 500 == 0:
            print_mean_rewards(bandits, total_rewards)

        i += 1

    # Check what the network thinks is the optimal action
    for a in range(bandits.num_bandits):
        print("The agent thinks action {} for bandit {} is the best".format(
            str(np.argmax(ww[a]) + 1), str(a + 1)
        ))

        if np.argmax(ww[a]) == np.argmin(bandits.bandits[a]):
            print("...and it was right!")
        else:
            print("...and it was wrong!")
