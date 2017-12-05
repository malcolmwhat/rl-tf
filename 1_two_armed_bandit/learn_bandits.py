# -*- coding: UTF-8 -*-

"""
This module is based on tutorial 1.

The purpose is to solve the n-armed bandit problem using
policy gradient and e-greedy action selection.
"""

import numpy as np
import tensorflow as tf
import numpy.random as rn


"""Band of bandits."""
band = {
    "bandits": [],
    "size": 0
}


def add_bandit(bandit_value):
    """Basic function for adding bandits."""
    band["bandits"].append(bandit_value)
    band["size"] = len(band["bandits"])


def pull_bandit(n):
    """Pull the n-th bandit.

    Return a reward of 1 if our random number is greater than the bandit value.
    Return a reward of -1 otherwise.
    """
    reward = 1 if np.random.randn(1) > band["bandits"][n] else -1
    return reward


class Agent(object):
    """The agent which acts in our n-armed bandit environment."""
    def __init__(self):
        tf.reset_default_graph()

        # Feed forward components.
        self.weights = tf.Variable(tf.ones([band["size"]]))
        self.chosen_action = tf.argmax(self.weights, 0)

        # Training procedure.
        self.reward_holder = tf.placeholder(shape=[1], dtype=tf.float32)
        self.action_holder = tf.placeholder(shape=[1], dtype=tf.int32)
        self.relevant_weight = tf.slice(self.weights, self.action_holder, [1])
        self.loss = -(tf.log(self.relevant_weight) * self.reward_holder)
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        self.update = self.optimizer.minimize(self.loss)

        self.init = tf.initialize_all_variables()
        self.session = None
        self.bandit_weights = None

    def start(self):
        self.session = tf.Session()
        self.session.run(self.init)

    def kill(self):
        self.session.close()

    def select_action(self, e):
        """Make e-greedy action selection.i

        Args
            e: The probability of selecting a random action.
        """
        if rn.rand(1) < e:
            return rn.randint(band["size"])
        else:
            # Infer the best action by picking dominant neural net weight.
            return self.session.run(self.chosen_action)

    def learn(self, action, reward):
        """Update the neural network based on the action, reward pair."""
        dict_in = {self.reward_holder: [reward],
                   self.action_holder: [action]}

        _, _, self.bandit_weights = self.session.run([self.update,
                                                     self.relevant_weight,
                                                     self.weights],
                                                     feed_dict=dict_in)


def run(episodes, bandits, e):
    """Explore the bandits with an agent!"""
    for bandit in bandits:
        add_bandit(bandit)

    total_rewards = np.zeros(band["size"])

    agent = Agent()
    agent.start()

    for i in range(episodes):
        action = agent.select_action(e)
        reward = pull_bandit(action)

        agent.learn(action, reward)

        total_rewards[action] += reward

        # Display the rewards as we progress.
        if i % 50 == 0:
            print("Reward for the {} bandits: {}".format(str(band["size"]),
                                                         str(total_rewards)))

    print("The agent thinks bandit {} is the best... \nThis is:".format(
                str(np.argmax(agent.bandit_weights) + 1)))

    print(np.argmax(agent.bandit_weights) == np.argmax(-np.array(band["bandits"])))

    agent.kill()


if __name__ == "__main__":
    run(1000, [0.2, 0, -0.2, -5], 0.1)
