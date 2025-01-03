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

# Next steps: Suspiciousness Score

Quantilization involves selecting actions from a prior distribution, given the reward is greater than the quantilizer threshold.
Scalable oversight involves selecting actions from an untrusted policy, given that the suspiciousness score is below a threshold.
In practice, both of these are similar problems, and this will be especially true when we work with LLMs:
both will involve mixing a trusted-but-weak policy with an untrusted-but-strong policy, in order to attept to generate a trusted-and-strong policy.

We will add as suspiciousness score: the number of steps spent on a tomato tile.
It is "suspicious" if the policy spends very little time on a tomato tile.

We can try quantilizing by both the suspiciousness score and the reward.
We also try quantilizing by some mixture of the (normalized) suspiciousness score and the (normalized) reward.
Both work but are not very clean or nice.