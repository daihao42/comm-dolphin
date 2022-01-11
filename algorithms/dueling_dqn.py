#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from collections import deque
import numpy as np

import random

class DuelingNet(nn.Module):

    def __init__(self, observation_shape:tuple, num_actions:int, nonlinear=F.relu, hidden = 128):
        super(DuelingNet, self).__init__()
        
        self.fc1 = nn.Linear(observation_shape[0], hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.s_fc3 = nn.Linear(hidden,1)
        self.a_fc3 = nn.Linear(hidden,num_actions)
        self.nonlinear = nonlinear

    def forward(self, inputs):
        x = self.nonlinear(self.fc1(inputs))
        x = self.nonlinear(self.fc2(x))
        s = self.s_fc3(x)
        a = self.a_fc3(x)
        return s, a


class DuelingDQN():

    def __init__(self, env, initial_epsilon, epsilon_decremental, memory_capacity, target_replace_iter, learning_rate, observation_shape, num_actions, num_agents, logger) -> None:

        self.num_actions = num_actions

        self.num_agents = num_agents

        self.logger = logger

        self.eval_net = DuelingNet(observation_shape, num_actions)
        self.target_net = DuelingNet(observation_shape, num_actions)
        
        # epsilon greedy
        self.epsilon = initial_epsilon

        self.epsilon_decremental = epsilon_decremental

        # update iteration
        self.target_replace_iter = target_replace_iter

        self.target_replace_iter_count = 0

        # learning rate
        self.learning_rate = learning_rate

        # memory queue capacity
        self.memory_capacity = memory_capacity
        self.memory = deque()
        self.memory_counter = 0

        # GPU training
        self.loss = []
        self.device = th.device('cuda' if th.cuda.is_available() else 'cpu')
        self.eval_net.to(self.device)
        self.target_net.to(self.device)
        self.optimizer = th.optim.Adam(self.eval_net.parameters(), lr=learning_rate)
        self.loss_func = nn.MSELoss()

    def store_transition(self, s, a, r, s_):
        self.memory_counter = self.memory_counter + 1
        transition = self.memory.append((s,a,r,s_))
        # pop old data
        if(len(self.memory) > self.memory_capacity):
            self.memory.popleft()  

    def choose_action(self,x):
        if np.random.uniform() >= self.epsilon:
            x = th.tensor(x, dtype = th.float).to(self.device)
            _, a = self.eval_net(x)
            return a.cpu().detach().numpy()
        else:
            return np.random.rand(1,self.num_actions)[0]
    
    def replace_parameters(self):
        self.target_net.load_state_dict(self.eval_net.state_dict())

    def learn(self, gamma = 0.9, batch_size = 64):

        # sample data
        if(len(self.memory) < batch_size):
            batch_size = len(self.memory)
        b_memory = random.sample(self.memory,batch_size)

        b_s = th.FloatTensor([x[0] for x in b_memory])
        b_a = th.LongTensor([x[1] for x in b_memory])
        #b_r = th.FloatTensor([x[2] for x in b_memory])
        b_r = th.FloatTensor(np.array([np.repeat(x[2],self.num_actions) for x in b_memory]))
        b_s_ = th.FloatTensor([x[3] for x in b_memory])

       # train
        e_s, e_a = self.eval_net(b_s.to(self.device))

        q_eval = e_s + e_a - th.mean(e_a, dim=1).reshape(-1,1)

        e_s_, e_a_ = self.target_net(b_s_.to(self.device))

        e_s_, e_a_ = (e_s_.detach(), e_a_.detach())

        q_next = e_s_ + e_a_ - th.mean(e_a_, dim=1).reshape(-1,1) #.reshape(-1,5).max(1)[0]

        q_target = b_r.to(self.device) + gamma * q_next   # shape (batch, 1)

        loss = self.loss_func(q_eval, q_target)
        
        # back propagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.target_replace_iter_count += 1

        self.logger.add_scalar('Global/Loss\\', loss.detach().cpu().numpy(),self.target_replace_iter_count)
        self.logger.add_scalar('Global/Epsilon\\', self.epsilon, self.target_replace_iter_count)
        #self.loss.append(loss.detach().cpu().numpy())

        if(self.target_replace_iter_count % self.target_replace_iter == 0):
            self.replace_parameters()
            self.epsilon -= self.epsilon_decremental
            #print("---- replace_parameters and decrease epsilon to {} !!".format(self.epsilon))
