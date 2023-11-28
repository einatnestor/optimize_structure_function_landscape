from torch import optim
import landscape
import dqn_pytorch_discrete as DQN
import numpy as np
import torch.nn as nn
import torch
import sys
from collections import namedtuple, deque
import random
from datetime import datetime
import math
import os
import copy
import pandas as pd
import pickle

SAVING_PERIOD = 10
ACTION_PER_SPECIES = 2
NUM_OF_ITERATIONS = 30
BATCH_SIZE = 64
GAMMA = 0.99
EPS_START = 1
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


def save_run(episodes, train_agent, environment, param, reward_arr, prev_agents):
    path = "./results/" + str(environment.num_species) + "_dqn_species_" + str(
        environment.action_per_species) + "_actions_" + str(
        episodes) + "_episodes_" + str(
        datetime.now().strftime("%m_%d_%Y_%H_%M"))
    os.mkdir(path)
    agent_pkl = open(path + '/agent.pkl', 'wb')
    pickle.dump(train_agent, agent_pkl)

    pd.DataFrame(environment.r_arr).to_csv(path + "/r_values.csv")
    pd.DataFrame(environment.alpha_arr).to_csv(path + "/alpha_values.csv")
    pd.DataFrame(environment.basel_production).to_csv(path + "/basel_production.csv")
    pd.DataFrame(environment.interaction_production).to_csv(path + "/interaction_production.csv")
    pd.DataFrame(reward_arr).to_csv(path + "/reward_per_episode.csv")

    params = open(path + '/params.pkl', 'wb')
    pickle.dump(param, params)

    os.mkdir(path + "/prev_agents")
    episode = SAVING_PERIOD
    for agent in prev_agents:
        prev_agent_pkl = open(path + '/prev_agents/' + str(episode) + '.pkl', 'wb')
        pickle.dump(agent, prev_agent_pkl)
        episode += SAVING_PERIOD


def train(env, params):
    # We get the shape of a state and the actions space size
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agents = []

    # We define our agent
    policy_net = DQN.DQN(state_size, action_size)
    target_net = DQN.DQN(state_size, action_size)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    memory = ReplayMemory(10000)#10000
    steps_done = 0
    reward_arr = []
    run = 0
    update = 0
    for i_episode in range(params['episodes']):
        # Initialize the environment and get it's state
        state = env.reset()
        done = False
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        reward_total = 0

        while not done:
            action = select_action(state, policy_net, env, steps_done)
            steps_done += 1
            next_state, reward, done, _ = env.step(action.item())
            reward_total += reward
            reward = torch.reshape(torch.tensor([reward]), [1, ])

            # Store the transition in memory
            a = env.simulate_dynamics(state)
            next_s = env.simulate_dynamics([next_state])
            next = torch.tensor(env.simulate_dynamics([next_state]))
            if done:
                 next_state = None
            memory.push(torch.reshape(torch.tensor(state, dtype=torch.float), [state_size, ]), action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            optimize_model(policy_net, target_net, optimizer, memory, state_size, run)
            # if loss != None:
            #     iter +=1
            #     loss_avg +=loss
            #
            # if done == True:
            #     if iter != 0:
            #         loss_avg = loss_avg/iter
            #         loss_avg_arr.append(loss_avg)


        update += 1
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
        target_net.load_state_dict(target_net_state_dict)
        # if update == 500:
        #     # Hard update of the target network's weights
        #     # θ′ ← τ θ + (1 −τ )θ′
        #     target_net_state_dict = target_net.state_dict()
        #     policy_net_state_dict = policy_net.state_dict()
        #     for key in policy_net_state_dict:
        #         target_net_state_dict[key] = policy_net_state_dict[key]
        #     target_net.load_state_dict(target_net_state_dict)
        #     update = 0


        run+=1
        if run == SAVING_PERIOD:
            start = int(BATCH_SIZE/NUM_OF_ITERATIONS)
            print("finish episode:" + str(i_episode) + " reward: " + str(reward_total))
            run = 0
            agents.append(copy.deepcopy(target_net))
        reward_arr.append(reward_total)

    save_run(params['episodes'], target_net, env, params, reward_arr, agents)
    a = 0


def select_action(state, policy_net, env, steps_done):
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            # a = torch.tensor(state,  dtype=torch.float)
            # b = policy_net(torch.tensor(state,  dtype=torch.float))
            # c = torch.reshape(torch.argmax(policy_net(torch.tensor(state,  dtype=torch.float))),[1,])
            return torch.reshape(torch.argmax(policy_net(torch.tensor(state, dtype=torch.float))), [1, ])
            # return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.reshape(torch.tensor([env.action_space.sample()], dtype=torch.long), [1, ])


def optimize_model(policy_net, target_net, optimizer, memory, state_size, run):
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), dtype=torch.bool)
    non_final_next_states = torch.tensor([s for s in batch.next_state if s is not None],dtype=torch.float)
    # non_final_next_states = torch.reshape(torch.cat([s for s in torch.tensor(batch.next_state, dtype=torch.float)
    #                                                  if s is not None]), (BATCH_SIZE, state_size))

    state_batch = torch.reshape(torch.cat(batch.state), (BATCH_SIZE, state_size))
    action_batch = torch.reshape(torch.cat(batch.action), (BATCH_SIZE, 1))
    reward_batch = torch.reshape(torch.cat(batch.reward), (BATCH_SIZE, ))

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    a  = policy_net(state_batch)
    b = a.gather(1, action_batch)
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE)
    with torch.no_grad():
        q = target_net(non_final_next_states).max(1)[0]
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    # Compute the expected Q values

    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    q=  torch.reshape(expected_state_action_values,(BATCH_SIZE, 1))
    loss = criterion(state_action_values, torch.reshape(expected_state_action_values,(BATCH_SIZE, 1)))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    #torch.nn.utils.clip_grad.clip_grad_norm_(policy_net.parameters(), 10)
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()
    return





def main():
    species_number, episodes = int(sys.argv[1]), int(sys.argv[2])
    params = {'alpha': 0.1, 'gamma': 0.9, 'epsilon': 1, 'episodes': episodes, "inner_iterations": NUM_OF_ITERATIONS}
    env = landscape.landscape(species_number, ACTION_PER_SPECIES, NUM_OF_ITERATIONS)
    train(env, params)


main()
