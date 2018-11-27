import torch, numpy, gym, random
import torch.nn as nn
import torch.optim as opt
from torch.distributions import Categorical
import matplotlib.pyplot as plt
from gym.envs.registration import register

register(
    id='FrozenLakeDeterministic8x8-v0',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name' : '8x8', 'is_slippery': False}
)

env = gym.make("FrozenLakeDeterministic8x8-v0")
action_space = env.action_space.n
observation_space = env.observation_space.n
network = nn.Sequential(nn.Linear(observation_space,4),nn.ReLU(),nn.Linear(4,action_space),nn.Softmax(dim=-1))
optimizer = opt.SGD(network.parameters(),lr=0.05)

epochs = 100000
discount_factor = 0.99

log_actions = torch.tensor([])
rewards = []

#Samples From Distribution(Softmax Output)
def choose_action(state):
    global log_actions
    state = torch.tensor(numpy.eye(observation_space)[state]).float()
    out = network(state)
    c = Categorical(out)
    action = c.sample()
    log = torch.tensor([0.0],requires_grad=True).add(c.log_prob(action))#torch.tensor([0.0],requires_grad=True)+torch.log(out[action])#torch.tensor([c.log_prob(action)],requires_grad=True)
    if len(log_actions.data)!=0:
        log_actions = torch.cat([log_actions,log])
    else:
        log_actions = log
    return action.item()

def maxprob(state):
    state = torch.tensor(numpy.eye(observation_space)[state]).float()
    prob = network(state)
    return prob.max(0)

def learn(loss):
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

def discount_rewards():
    sum = 0
    discounted_rewards = []
    for r in rewards[::-1]:
        sum = sum * discount_factor + r
        discounted_rewards.insert(0, sum)
    return discounted_rewards

def update_policy():
    global rewards, log_actions
    rewards = torch.tensor(discount_rewards()).float()
    #rewards = (rewards - rewards.mean()) / (rewards.std() + numpy.finfo(float).eps)
    loss = torch.sum(torch.mul(rewards,log_actions).mul(-1),-1)
    learn(loss)
    rewards=[]
    log_actions = torch.tensor([])

print("---------TRAINING---------")
allrewards = []
rewardmeans = []
for e in range(epochs):
        frames = 0
        runreward = 0
        state = env.reset()

        while True:
            frames += 1
            action = choose_action(state)
            new_state, reward, done, i = env.step(action)

            runreward+=reward
            rewards.append(reward)
            state = new_state
            if done or frames%40 == 0:
                update_policy()
                #print("E: "+str(e)+" F: "+str(frames)+" R: "+str(runreward))
                allrewards.append(runreward)
                break

        if (e+1)%100==0:
            mean = numpy.mean(allrewards[-100:])
            print(str(e + 1) + " M: " + str(mean))
            rewardmeans.append(mean)
            if mean == 1.0:
                print("SOLVED!!!")
                break

print("---------TESTING---------")
for e in range(1):
        state = env.reset()

        while True:
            prob, action = maxprob(state)
            new_state, reward, done, i = env.step(action.item())
            env.render()
            state = new_state
            if done or frames%40 == 0:
                print("E: "+str(e)+" F: "+str(frames)+" R: "+str(runreward))
                break

print("---------PLOTTING---------")
plt.figure(0)
plt.plot(allrewards)
plt.title("Reward")
#plt.savefig('./Plots/PolicyGradient/8x8det/reward.png',bbox_inches='tight')
plt.figure(1)
plt.plot(rewardmeans,color="orange")
plt.title("Mean Reward")
#plt.savefig('./Plots/PolicyGradient/8x8det/meanreward.png',bbox_inches='tight')
plt.show()