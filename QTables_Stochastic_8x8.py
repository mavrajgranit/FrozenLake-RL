import gym, numpy,random
import matplotlib.pyplot as plt
from gym.envs.registration import register

register(
    id='FrozenLakeDeterministic8x8-v0',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name' : '8x8', 'is_slippery': True}
)

env = gym.make("FrozenLakeDeterministic8x8-v0")
action_space = env.action_space.n
observation_space = env.observation_space.n
qtable = numpy.zeros((observation_space,action_space))

epochs = 30000
max_epsilon = 1.0
min_epsilon = 0.0
percent = 0.35
decay_rate = 1/(epochs*percent)
epsilon = max_epsilon

discount_factor = 0.98
learning_rate = 0.4

def eps_greedy(state):
    row = qtable[state]

    if random.random()>=epsilon:
        action = numpy.argmax(row)
        return row[action],action
    else:
        action = random.randint(0,action_space-1)
        return row[action],action

def maxq(state):
    row = qtable[state]
    return numpy.max(row)

def decay():
    return max(min_epsilon,epsilon-decay_rate)

print("---------TRAINING---------")
rewards = []
scoremeans = []
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

            #Force it to reach the goal faster
            #reward = -1 if done and reward != 1 else reward

            nt = int(not done)
            nq = maxq(new_state)
            qtable[state][action] += learning_rate*( (nt*discount_factor*nq + reward) - q)

            runreward+=reward
            state = new_state
            if done or frames%100 == 0:
                #print("E: "+str(e)+" F: "+str(frames)+" R: "+str(runreward)+" Eps: "+str(epsilon))
                rewards.append(runreward)
                qvals += qval
                break

        epsilon = decay()
        if (e+1)%100==0:
            mean = numpy.mean(rewards[-100:])
            print(str(e+1)+" M: "+str(mean))
            scoremeans.append(mean)
            qvalues.append(qvals/100)
            qvals = 0
            if mean>=0.78:
                print("SOLVED!!!")
                break

print("---------TESTING---------")
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
            if done or frames%100 == 0:
                print(qs)
                #print("E: "+str(e)+" F: "+str(frames)+" R: "+str(runreward)+" Eps: "+str(epsilon))
                break
print("---------Q_TABLE---------")
print(qtable)

print("---------PLOTTING---------")
plt.figure(0)
plt.plot(rewards)
plt.title("Reward")
#plt.savefig('./Plots/QTables/8x8sto/reward.png',bbox_inches='tight')
plt.figure(1)
plt.plot(scoremeans,color="orange")
plt.title("Mean Reward")
#plt.savefig('./Plots/QTables/8x8sto/meanreward.png',bbox_inches='tight')
plt.figure(2)
plt.plot(qvalues,color="red")
plt.title("Mean Q-Value")
#plt.savefig('./Plots/QTables/8x8sto/meanqvalue.png',bbox_inches='tight')
plt.show()