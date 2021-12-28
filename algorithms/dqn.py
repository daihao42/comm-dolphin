#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import deque
import numpy as np
import torch as th
import torch.nn as nn

import random

from torch.autograd import Variable

__author__ = 'dai'

class DQN():
    def __init__(self, env, intial_epsilon, memory_capacity, target_replace_iter, learning_rate, net) -> None:
        # epsilon greedy
        self.epsilon = intial_epsilon

        # update iteration
        self.target_replace_iter = target_replace_iter

        # learning rate
        self.learning_rate = learning_rate

        # memory queue capacity
        self.memory_capacity = memory_capacity
        self.memory = deque()
        self.memory_counter = 0

        self.net = net

        # GPU training
        self.loss = []
        self.device = th.device('cuda' if th.cuda.is_available() else 'cpu')
        self.net.to(self.device)
        self.optimizer = th.optim.Adam(self.net.parameters(), lr=learning_rate)
        self.loss_func = nn.MSELoss()

    def store_transition(self, s, a, r, s_):
        self.memory_counter = self.memory_counter + 1
        transition = self.memory.append((s,a,r,s_))
        # pop old data
        if(len(self.memory) > self.memory_capacity):
            self.memory.popleft()  

    def choose_action(self,x):
        x = th.tensor(x, dtype = th.float).to(self.device)
        return self.net(x).cpu()

    def learn(self, gamma = 0.5, batch_size = 32):

        # sample data
        if(len(self.memory) < batch_size):
            batch_size = len(self.memory)
        b_memory = random.sample(self.memory,batch_size)

        b_s = th.FloatTensor([x[0] for x in b_memory])
        b_a = th.LongTensor([x[1] for x in b_memory])
        b_r = th.FloatTensor([x[2] for x in b_memory])
        b_s_ = th.FloatTensor([x[3] for x in b_memory])

        # train
        q_eval = self.net(b_s.to(self.device)).max(1)[0] # shape (batch, 1)
        q_next = self.net(b_s_.to(self.device)).detach().max(1)[0] # detach from graph, don't backpropagate
        q_target = b_r.to(self.device) + gamma * q_next   # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)
        
        # back propagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.loss.append(loss.detach().cpu().numpy())
