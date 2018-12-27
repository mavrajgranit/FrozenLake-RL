# FrozenLake - Reinforcement Learning
Aims to solve FrozenLake with various RL-Techniques at a beginner level.

## Deterministic
- [x] Q-Tables
- [x] Deep Q Learning(no fixed Targets, no Memory, no Double-Q)
- [x] Policy Gradient

The remaining techniques won't be tested in this simple case.

## Stochastic
To be researched in depth.
Maybe N-Step Q-Learning could solve it?
How about trying to solve it with Dynamic Programming techniques such as Policy Evaluation/Iteration or Value Iteration?(Missing action probability distribution)

# Results 
Exploration and Learning Rate are big factors of convergence-time and -probability 

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

## Q-Learning Deterministic 4x4
<p float="left">
  <img src="./Plots/QLearning/4x4det/reward.png" width="425" />
  <img src="./Plots/QLearning/4x4det/meanreward.png" width="425" /> 
  <img src="./Plots/QLearning/4x4det/meanqvalue.png" width="425" />
</p>

## Q-Learning Deterministic 8x8
<p float="left">
  <img src="./Plots/QLearning/8x8det/reward.png" width="425" />
  <img src="./Plots/QLearning/8x8det/meanreward.png" width="425" /> 
  <img src="./Plots/QLearning/8x8det/meanqvalue.png" width="425" />
</p>

## PolicyGradient Deterministic 4x4
<p float="left">
  <img src="./Plots/PolicyGradient/4x4det/reward.png" width="425" />
  <img src="./Plots/PolicyGradient/4x4det/meanreward.png" width="425" /> 
</p>

## PolicyGradient Deterministic 8x8
Tends to fall to zero inbetween very high to mid results
<p float="left">
  <img src="./Plots/PolicyGradient/8x8det/reward.png" width="425" />
  <img src="./Plots/PolicyGradient/8x8det/meanreward.png" width="425" /> 
</p>
