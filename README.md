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