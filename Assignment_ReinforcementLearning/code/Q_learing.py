'''
 * @file Q_learing.py
 * @brief The Agent for implementing the Q-learning method and SARSA method
 *        to the box-game, then we have functions to output curves and graphs.
 *
 * [AI 6101] Introduction to AI & AI Ethics
 *           Course Assignment 2021
 *
 * @author Hantao Li  <hli038@e.ntu.edu.sg>
 *
 * @date 2021-10-13
 *
 *************************************************************************
 * VERSION FOR Q-LEARNING AND SARSA
 *************************************************************************
 *
 * Map struct ->  BoxPushing grid-world game
 *          ┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐
 *          │ 0 │ 1 │ 2 │ 3 │ 4 │ 5 │ 6 │ 7 │ 8 │ 9 │1 0│1 1│1 2│1 3│
 *      ┌───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
 *      │ 0 │   │   │   │   │   │   │   │   │   │   │   │   │   │   │
 *      ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
 *      │ 1 │   │   │   │   │   │   │   │   │   │   │   │   │   │   │
 *      ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
 *      │ 2 │   │   │   │   │   │   │   │   │   │   │   │   │   │   │
 *      ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
 *      │ 3 │   │   │   │   │   │   │   │   │   │   │   │   │   │   │
 *      ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
 *      │ 4 │   │ B │   │▓▓▓│▓▓▓│▓▓▓│▓▓▓│▓▓▓│▓▓▓│▓▓▓│▓▓▓│▓▓▓│ G │   │
 *      ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
 *      │ 5 │ A │   │▓▓▓│▓▓▓│▓▓▓│▓▓▓│▓▓▓│▓▓▓│▓▓▓│▓▓▓│▓▓▓│▓▓▓│▓▓▓│   │
 *      └───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┘
 *      A->Agent, B->Box, G->Goal, ▓▓▓->Lava
 *
 * RL Algorithms ->
 *      1. Q-Learning Algorithm
 *      2. SARSA Algorithm
 *
 *************************************************************************
'''

from environment import CliffBoxPushingBase
from collections import defaultdict
import numpy as np
import pandas as pd
import random
import math
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

'''
********************************************
*      GRAPGH DRAWING FUNCTIONS BEGIN      *
********************************************
'''


def DrawAboveLine():
    print('    ┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐')
    print('    │ 0 │ 1 │ 2 │ 3 │ 4 │ 5 │ 6 │ 7 │ 8 │ 9 │1 0│1 1│1 2│1 3│')
    print('┌───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤')


def DrawMiddleLine():
    print('├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤')


def DrawBelowLine():
    print('└───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┘')


def GetActionArrow(Q, raw, col, box_loc):
    '''
    Function for extract the max Q-value in a specific Q-table location
    Then show the action with the arrow representation

    Parameters
    ----------
    Q : this is the Q-table we use
    raw, col : these are the target agent block location, the index in Q-table
    box_loc : this is the box location, which is also the index in Q-table

    Returns
    -------
    arr : str
        the arrow showing the action in Q-table

    '''
    state = ((raw, col), box_loc)
    if Q[state].any() == 0:
        arr = ' '
    else:
        arrows = ['↑', '↓', '←', '→']
        arr = arrows[np.argmax(Q[state])]
    return arr


def DrawBlock(Q, raw, col, agent_loc, box_loc, goal_loc):
    '''
    Function for draw the specific block in the map
    Use A->Agent, B->Box, G->Goal, ▓▓▓->Lava, and Arrows for Q value-actions.
     
    Parameters
    ----------
     Q : this is the Q-table we use
     raw, col : these are the target block location
     agent_loc, box_loc, goal_loc - these are the map data
    '''
    if agent_loc == (raw, col):
        print(' A │', end='')
    elif box_loc == (raw, col):
        print(' B │', end='')
    elif goal_loc == (raw, col):
        print(' G │', end='')
    elif (raw, col) in lava_loc:
        print('▓▓▓│', end='')
    else:
        arr = GetActionArrow(Q, raw, col, box_loc)
        print(' ' + arr + ' │', end='')


def DrawEmptyBlock(raw, col, agent_loc, box_loc, goal_loc):
    '''
    The another version of function for draw the specific block in the map
    Use A->Agent, B->Box, G->Goal, ▓▓▓->Lava, 
    Here, we don't fill the block with arrows but leave them empty.
    '''
    if agent_loc == (raw, col):
        print(' A │', end='')
    elif box_loc == (raw, col):
        print(' B │', end='')
    elif goal_loc == (raw, col):
        print(' G │', end='')
    elif (raw, col) in lava_loc:
        print('▓▓▓│', end='')
    else:
        print('   │', end='')


def DrawMap(Q, agent_loc, box_loc, goal_loc):
    '''
    Function for draw the whole map with the states currently.
    Use A->Agent, B->Box, G->Goal, ▓▓▓->Lava, and Arrows for Q value-actions.
     
    Parameters
    ----------
     Q : this is the Q-table we use
     agent_loc, box_loc, goal_loc : these are the map data
    '''
    DrawAboveLine()
    for raw in range(6):
        print('│ ' + str(raw) + ' │', end='')
        for col in range(14):
            DrawBlock(Q, raw, col, agent_loc, box_loc, goal_loc)
            if col == 13:
                print('')
        if raw == 5:
            DrawBelowLine()
        else:
            DrawMiddleLine()
    action_cur = GetActionArrow(Q, agent_loc[0], agent_loc[1], box_loc)
    print('Agent Action Now: ' + action_cur)


def DrawEmptyMap(agent_loc, box_loc, goal_loc):
    '''
    The another version of function for draw the whole map
    Use A->Agent, B->Box, G->Goal, ▓▓▓->Lava, 
    Here, we don't fill the block with arrows but leave them empty.
    '''
    DrawAboveLine()
    for raw in range(6):
        print('│ ' + str(raw) + ' │', end='')
        for col in range(14):
            DrawEmptyBlock(raw, col, agent_loc, box_loc, goal_loc)
            if col == 13:
                print('')
        if raw == 5:
            DrawBelowLine()
        else:
            DrawMiddleLine()


'''
********************************************
*      RL ALGORITHMS BEGIN                 *
********************************************
'''

class QAgent(object):
    def __init__(self):
        self.agent_name = 'QLearning'
        self.action_space = [1, 2, 3, 4]
        self.Q = defaultdict(lambda: np.zeros(len(self.action_space)))
        self.discount_factor = 0.99
        self.alpha = 0.5
        self.epsilon = 0.01

    def take_action(self, state):
        if random.random() < self.epsilon:
            action = random.choice(self.action_space)
        else:
            action = self.action_space[np.argmax(self.Q[state])]
        return action

    def train(self, state, action, next_state, reward):
        self.Q[state][action-1] = (1 - self.alpha) * self.Q[state][action-1] \
            + self.alpha * (reward + self.discount_factor \
                            * max(self.Q[next_state]))
        pass


class SARSA_Agent(object):
    def __init__(self):
        self.agent_name = 'SARSA'
        self.action_space = [1, 2, 3, 4]
        self.Q = defaultdict(lambda: np.zeros(len(self.action_space)))
        self.discount_factor = 0.99
        self.alpha = 0.5
        self.epsilon = 0.01

    def take_action(self, state):
        if random.random() < self.epsilon:
            action = random.choice(self.action_space)
        else:
            action = self.action_space[np.argmax(self.Q[state])]
        return action

    def train(self, state, action, next_action, next_state, reward):
        self.Q[state][action-1] = (1 - self.alpha) * self.Q[state][action-1] \
            + self.alpha * (reward + self.discount_factor \
                            * self.Q[next_state][next_action-1])
        pass


'''
********************************************
*      MAIN FUNCTIONS BEGIN                *
********************************************
'''

#Load the map of lava
lava_loc = []
for i in range(2, 13):
    lava_loc.append((5, i))
for i in range(3, 12):
    lava_loc.append((4, i))

#Function for the consine annealing
def CosineAnnealing(epsilon, t, T):
    return(0.5*epsilon*(1+np.cos((t/T)*math.pi)))

if __name__ == '__main__':
    reward_all = []
    qvalue_all = []
    loop = 20  #number of loop for calculate the average curve
    inital_epsilon = 0.01  #inital epsilon-greedy for the agent
    for loop_time in range(loop):
        env = CliffBoxPushingBase()
        teminated = False
        rewards = []
        rewards_record = []
        time_step = 0
        record_qvalue = [[], [], [], []]

        #choose the agent and parameters here:
        agent = QAgent()
        #agent = SARSA_Agent()
        agent.epsilon = inital_epsilon
        num_iterations = 10000  #number of episodes
        qvalue_loc = ((4, 1), (3, 1)) #the state of the saving q-value
        
        if agent.agent_name == 'QLearning':
            for i in range(num_iterations):
                agent.epsilon = CosineAnnealing(inital_epsilon, i, num_iterations)
                env.reset()
                while not teminated:
                    state = env.get_state()
                    action = agent.take_action(state)
                    #print(action)
                    reward, teminated, _ = env.step([action])
                    next_state = env.get_state()
                    rewards.append(reward)
                    #print(f'step: {time_step}, actions: {action}, reward: {reward}')
                    time_step += 1
                    agent.train(state, action, next_state, reward)
                #print(f'rewards: {sum(rewards)}')
                #print(f'print the historical actions: {env.episode_actions}')
                teminated = False
                if (i % 10) == 0:
                    rewards_record.append(sum(rewards))
                    record_qvalue[0].append(agent.Q[qvalue_loc][0])
                    record_qvalue[1].append(agent.Q[qvalue_loc][1])
                    record_qvalue[2].append(agent.Q[qvalue_loc][2])
                    record_qvalue[3].append(agent.Q[qvalue_loc][3])
                rewards = []

        elif agent.agent_name == 'SARSA':
            for i in range(num_iterations):
                agent.epsilon = CosineAnnealing(inital_epsilon, i, num_iterations)
                env.reset()
                while not teminated:
                    state = env.get_state()
                    if rewards == []:
                        next_action = agent.take_action(state)
                    action = next_action
                    #print(action)
                    reward, teminated, _ = env.step([action])
                    next_state = env.get_state()
                    rewards.append(reward)
                    #print(f'step: {time_step}, actions: {action}, reward: {reward}')
                    time_step += 1
                    next_action = agent.take_action(next_state)
                    agent.train(state, action, next_action, next_state, reward)
                #print(f'rewards: {sum(rewards)}')
                #print(f'print the historical actions: {env.episode_actions}')
                teminated = False
                if (i % 10) == 0:
                    rewards_record.append(sum(rewards))
                    record_qvalue[0].append(agent.Q[qvalue_loc][0])
                    record_qvalue[1].append(agent.Q[qvalue_loc][1])
                    record_qvalue[2].append(agent.Q[qvalue_loc][2])
                    record_qvalue[3].append(agent.Q[qvalue_loc][3])
                rewards = []
                
        print('Finished for Loop ' + str(loop_time + 1) + ' ...')
        reward_all.append(rewards_record)
        qvalue_all.append(record_qvalue)
        
        
'''
********************************************
*      OUTPUTS PARTS BEGIN                 *
********************************************
(Usually, we use another script/Jupyter to obtain the output images; thus, 
 we only show some examples of outputing parts here)
'''

#Draw the map of the game with Q-value showing
DrawMap(agent.Q, (1,11), (2,12), (4,12))

#Draw the curve of episodes vs. rewards
reward_x = np.vstack((reward_all[0], reward_all[1]))
for i in range(2, loop):
    reward_x = np.vstack((reward_x, reward_all[i]))
df = pd.DataFrame(reward_x).melt(var_name='Episode/10',value_name='Rewards')
plt.figure(dpi=600)
sns.lineplot(x="Episode/10", y="Rewards", data=df)
plt.title('SARSA with Cosine Annealing Epsilon')
plt.savefig("./QL_0.01_cos.png", dpi=600)
plt.show()


#Draw the curve of episodes vs. Q-Value
def SortQValue(qvalue_all, loop):
    q_0 = np.vstack((qvalue_all[0][0], qvalue_all[1][0]))
    for i in range(2, loop):
        q_0 = np.vstack((q_0, qvalue_all[i][0]))
        
    q_1 = np.vstack((qvalue_all[0][1], qvalue_all[1][1]))
    for i in range(2, loop):
        q_0 = np.vstack((q_0, qvalue_all[i][1]))
        
    q_2 = np.vstack((qvalue_all[0][2], qvalue_all[1][2]))
    for i in range(2, loop):
        q_0 = np.vstack((q_0, qvalue_all[i][2]))
        
    q_3 = np.vstack((qvalue_all[0][3], qvalue_all[1][3]))
    for i in range(2, loop):
        q_0 = np.vstack((q_0, qvalue_all[i][3]))  
    return q_0, q_1, q_2, q_3

data = SortQValue(qvalue_all, loop)
label = ['UP', 'DOWN', 'LEFT', 'RIGHT']
df=[]
for i in range(len(data)):
    df.append(pd.DataFrame(data[i]).melt(var_name='Episodes',value_name='Q-Value'))
    df[i]['Action']= label[i] 

plt.figure(dpi=600)
df=pd.concat(df)
sns.lineplot(x="Episodes", y="Q-Value", hue="Action", data=df)
plt.title('Q-Value - SASAR - Cosine Annealing')
plt.savefig("./Q_QL_cos.png", dpi=600)
plt.show()
