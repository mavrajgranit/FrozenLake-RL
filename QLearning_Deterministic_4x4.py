import torch, numpy, gym, random
import torch.nn as nn
import torch.optim as opt
import matplotlib.pyplot as plt
from gym.envs.registration import register

register(
    id='FrozenLakeDeterministic4x4-v0',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name' : '4x4', 'is_slippery': False}
)

env = gym.make("FrozenLakeDeterministic4x4-v0")
action_space = env.action_space.n
observation_space = env.observation_space.n
network = nn.Sequential(nn.Linear(observation_space,4),nn.ReLU(),nn.Linear(4,action_space))
criterion = nn.MSELoss()
optimizer = opt.SGD(network.parameters(),lr=0.1)
#Lower Learning Rate And more Exploration makes Convergence more likely

epochs = 100000
max_epsilon = 1.0
min_epsilon = 0.01
percent = 0.05
decay_rate = 1/(epochs*percent)
epsilon = max_epsilon

discount_factor = 0.99


def eps_greedy(state):
    state = torch.tensor(numpy.eye(observation_space)[state]).float()
    qs = network(state)

    if random.random()>epsilon:
        q,action = qs.max(0)
        return qs,q,action.item()
    else:
        action = random.randint(0,action_space-1)
        return qs,qs[action],action

def maxq(state):
    state = torch.tensor(numpy.eye(observation_space)[state]).float()
    qs = network(state)
    return qs.max(0)

def decay():
    return max(min_epsilon,epsilon-decay_rate)

def learn(out,target):
    optimizer.zero_grad()
    loss = criterion(out,target)
    loss.backward()
    optimizer.step()
    return loss

print("---------TRAINING---------")
rewards = []
rewardmeans = []
qvalues = []
qvals = 0
for e in range(epochs):
        frames = 0
        runreward = 0
        state = env.reset()
        qval = 0

        while True:
            frames += 1
            qs,q, action = eps_greedy(state)
            qval += q.item()
            new_state, reward, done, i = env.step(action)
            nt = int(not done)

            target = qs.clone()
            nq,naction = maxq(new_state)
            target[action] = nt*discount_factor*nq + reward
            learn(qs,target)

            runreward+=reward
            state = new_state
            if done or frames%20 == 0:
                #print("E: "+str(e)+" F: "+str(frames)+" R: "+str(runreward)+" Eps: "+str(epsilon))
                rewards.append(runreward)
                qvals += qval
                break
        epsilon = decay()
        if (e+1)%100==0:
            mean = numpy.mean(rewards[-100:])
            qvalues.append(qvals / 100)
            qvals = 0
            print(str(e + 1) + " M: " + str(mean))
            rewardmeans.append(mean)
            if mean == 1.0:
                print("SOLVED!!!")
                break

print("---------TESTING---------")
for e in range(1):
        state = env.reset()
        qs = 0
        while True:
            qq, q, action = eps_greedy(state)
            qs+=q
            new_state, reward, done, i = env.step(action)
            env.render()
            state = new_state
            if done or frames%20 == 0:
                print(qs)
                #print("E: "+str(e)+" F: "+str(frames)+" R: "+str(runreward)+" Eps: "+str(epsilon))
                break

print("---------PSEUDO_Q_TABLE---------")
for t in range(observation_space):
    print(network(torch.tensor(numpy.eye(observation_space)[t]).float()))

print("---------PLOTTING---------")
plt.figure(0)
plt.plot(rewards)
plt.title("Reward")
#plt.savefig('./Plots/QLearning/4x4det/reward.png',bbox_inches='tight')
plt.figure(1)
plt.plot(rewardmeans,color="orange")
plt.title("Mean Reward")
#plt.savefig('./Plots/QLearning/4x4det/meanreward.png',bbox_inches='tight')
plt.figure(2)
plt.plot(qvalues,color="red")
plt.title("Mean Q-Value")
#plt.savefig('./Plots/QLearning/4x4det/meanqvalue.png',bbox_inches='tight')
plt.show()