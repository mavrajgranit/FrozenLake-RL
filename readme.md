# FrozenLake - Reinforcement Learning
Aims to solve FrozenLake with various RL-Techniques at a beginner level.

## Deterministic
- [x] Q-Tables
- [ ] Deep Q Learning(no fixed Targets, no Memory, no Double-Q)
- [ ] Policy Gradient

The remaining Techniques won't be tested in this simple case.

## Stochastic
To be researched in depth.
Current guess is estimating a distribution of actions instead of reward-values.
Maybe Policy Gradients or N-Step Q-Learning could solve it?

# Results
## Q-Tables Deterministic 4x4
<!-- ![qtablemeanreward](./Plots/QTables/meanreward.png) ![qtablemeanscore](./Plots/QTables/meanscore.png) ![qtableqvalues](./Plots/QTables/qvalues.png) -->

<p float="left">
  <img src="./Plots/QTables/4x4det/reward.png" width="425" />
  <img src="./Plots/QTables/4x4det/meanreward.png" width="425" /> 
  <img src="./Plots/QTables/4x4det/meanqvalue.png" width="425" />
</p>

## Q-Tables Deterministic 8x8
<p float="left">
  <img src="./Plots/QTables/8x8det/reward.png" width="425" />
  <img src="./Plots/QTables/8x8det/meanreward.png" width="425" /> 
  <img src="./Plots/QTables/8x8det/meanqvalue.png" width="425" />
</p>