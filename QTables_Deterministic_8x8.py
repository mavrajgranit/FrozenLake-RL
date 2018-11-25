import gym, numpy,random
import matplotlib.pyplot as plt
from gym.envs.registration import register

register(
    id='FrozenLakeDeterministic4x4-v0',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name' : '8x8', 'is_slippery': False}
)

env = gym.make("FrozenLakeDeterministic4x4-v0")
action_space = env.action_space.n
observation_space = env.observation_space.n
qtable = numpy.zeros((observation_space,action_space))
# Cant Get Stuck As Easily: qtable = numpy.random.random((observation_space,action_space))

epochs = 100000
max_epsilon = 1.0
min_epsilon = 0.01
#Random Epochs Percentage
percent = 0.4
decay_rate = 1/(epochs*percent)
epsilon = max_epsilon

#A Little Less Than 1 So Actions Dont Reach A Reward Of 1 If They Aren't The Last Action
discount_factor = 0.99
learning_rate = 0.9

#Follows Epsilon Greedy Action Selection
def eps_greedy(state):
    row = qtable[state]

    if random.random()>=epsilon:
        action = numpy.argmax(row)
        return row[action],action
    else:
        action = random.randint(0,action_space-1)
        return row[action],action

#Returns Highest Expected Reward Estimate For State
def maxq(state):
    row = qtable[state]
    return numpy.max(row)

#Linear Decay
def decay():
    return max(min_epsilon,epsilon-decay_rate)

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
            q, action = eps_greedy(state)
            qval += q.item()
            new_state, reward, done, i = env.step(action)

            nt = int(not done)
            nq = maxq(new_state)
            qtable[state][action] += learning_rate*( (nt*discount_factor*nq + reward) - q)

            runreward+=reward
            state = new_state
            if done or frames%40 == 0:
                #print("E: "+str(e)+" F: "+str(frames)+" R: "+str(runreward)+" Eps: "+str(epsilon))
                rewards.append(runreward)
                qvals += qval
                break

        epsilon = decay()
        if (e+1)%100==0:
            mean = numpy.mean(rewards[-100:])
            qvalues.append(qvals/100)
            qvals = 0
            print(str(e+1)+" M: "+str(mean))
            rewardmeans.append(mean)
            if mean==1.0:
                print("SOLVED!!!")
                break

print("---------TESTING---------")
#Tests with 1% randomness
for e in range(1):
        state = env.reset()
        qs = 0
        frames = 0
        while True:
            frames+=1

            q, action = eps_greedy(state)
            qs+=q
            new_state, reward, done, i = env.step(action)
            env.render()
            state = new_state
            if done or frames%40 == 0:
                print(qs)
                #print("E: "+str(e)+" F: "+str(frames)+" R: "+str(runreward)+" Eps: "+str(epsilon))
                break
print("---------Q_TABLE---------")
print(qtable)

print("---------PLOTTING---------")
plt.figure(0)
plt.plot(rewards)
plt.title("Reward")
plt.savefig('./Plots/QTables/8x8det/reward.png',bbox_inches='tight')
plt.figure(1)
plt.plot(rewardmeans,color="orange")
plt.title("Mean Reward")
plt.savefig('./Plots/QTables/8x8det/meanreward.png',bbox_inches='tight')
plt.figure(2)
plt.plot(qvalues,color="red")
plt.title("Mean Q-Value")
plt.savefig('./Plots/QTables/8x8det/meanqvalue.png',bbox_inches='tight')
plt.show()