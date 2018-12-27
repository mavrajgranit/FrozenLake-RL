# FrozenLake - Reinforcement Learning
Aims to solve FrozenLake with various RL-Techniques at a beginner level.

## [Deterministic](#det)
- [x] Q-Tables
- [x] Deep Q Learning(no fixed Targets, no Memory, no Double-Q)
- [x] Policy Gradient

The remaining techniques won't be tested in this simple case.

## [Stochastic](#sto)
- [x] Q-Tables
- [ ] Deep Q Learning
- [ ] Policy Gradient

To be researched in depth.
Current guess is estimating a distribution of actions instead of reward-values.
Maybe Policy Gradients or N-Step Q-Learning could solve it?

# Results 
Exploration and Learning Rate are big factors of convergence-time and -probability 

<a name="det"></a>
# Deterministic
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

<a name="sto"></a>
# Stochastic
## Q-Tables Stochastic 4x4
<!-- ![qtablemeanreward](./Plots/QTables/meanreward.png) ![qtablemeanscore](./Plots/QTables/meanscore.png) ![qtableqvalues](./Plots/QTables/qvalues.png) -->

<p float="left">
  <img src="./Plots/QTables/4x4sto/reward.png" width="425" />
  <img src="./Plots/QTables/4x4sto/meanreward.png" width="425" /> 
  <img src="./Plots/QTables/4x4sto/meanqvalue.png" width="425" />
</p>

## Q-Tables Stochastic 8x8
<p float="left">
  <img src="./Plots/QTables/8x8sto/reward.png" width="425" />
  <img src="./Plots/QTables/8x8sto/meanreward.png" width="425" /> 
  <img src="./Plots/QTables/8x8sto/meanqvalue.png" width="425" />
</p>