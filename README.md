This is a repository for the AISC 2025 SSO project on the tomatoes game.

utils.py and script.py, well guess what they do.

# Overview

We're looking at quantilizer-ish things on the tomato-watering gridworld scenario.
Our main considerations are how to generate the reference/prior policy distribution which we'll then quantilize according to misspecified reward.

A basic way is to choose a sequence of random actions from the set \{UP, DOWN, LEFT, RIGHT, WAIT\}, to make a random policy.
We also have code \(courtesy of Claude\) which uses KMV compression to generate a "simpler" policy as measured by compression ratio.

The next choice we have to make is whether to an invalid move is replaced by WAIT or a random valid move.

We find that quantilization **only** works well when we replace invalid moves with a random valid move **and** we do not use the KMV simplification method.

When the invalid move is replaced by WAIT, we find that the quantilization is ineffective.
If we use the KMV simplification method, we find that the quantilization is catastrophic.

# Q-Matrix Solving

We have code which uses dynamic programming to solve for a Q-matrix of all possible states and actions.
There are 2^13 * 29 * 5 = 1187840 entries in this matrix.
We are able to solve this through backward iteration on the Bellman equation.

We find that our attempts at capped soft Q-learning achieve better success than soft Q-learning,
though they compare poorly to a baseline of ideal performance.

# Random Policy Testing

Here we sample random action sequences from the set \{UP, DOWN, LEFT, RIGHT, WAIT\} and calculate the utility and misspecified reward.
We can threshold these by misspecified reward, then calculate the average reward and utility at each threshold.
These results are worse than the local quantilization algorithm.

# Q-Learning

Here we use a deep Q-learning algorithm to learn a policy based on SGD on finite data.
We aim to reproduce the performance of the Q-matrix solving system.