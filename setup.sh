#!/bin/bash
#
# Setup for Reinforcement learning with TF tutorials.
pushd ./..
git clone git@github.com:openai/gym.git

pushd gym
pip install -e .

popd
popd
