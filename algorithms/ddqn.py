#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import deque
import numpy as np
import torch as th
import torch.nn as nn

import random

from torch.autograd import Variable

from models.ActorNet import ActorNet

__author__ = 'dai'

class DDQN():
    def __init__(self, env, initial_epsilon, epsilon_decremental, memory_capacity, target_replace_iter, learning_rate, observation_shape, num_actions, num_agents, logger) -> None:

        self.num_actions = num_actions

        self.num_agents = num_agents

        self.logger = logger

        self.eval_net = ActorNet(observation_shape, num_actions)
        self.target_net = ActorNet(observation_shape, num_actions)
        
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

    def choose_action(self,x, exploration = True):
        if not exploration:
            x = th.tensor(x, dtype = th.float).to(self.device)
            return self.eval_net(x).cpu().detach().numpy()

        if np.random.uniform() >= self.epsilon:
            x = th.tensor(x, dtype = th.float).to(self.device)
            return self.eval_net(x).cpu().detach().numpy()
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
        b_r = th.FloatTensor([x[2] for x in b_memory])
        b_s_ = th.FloatTensor([x[3] for x in b_memory])

        '''
        
	odata[0]
	>>tensor([[ 0.2893, -0.0434,  0.0402, -0.0237,  0.0689],
		[ 0.2602, -0.0337,  0.0780, -0.1334, -0.0147],
		[-0.0032,  0.0756, -0.2571, -0.1774, -0.0014],
		[ 0.0097, -0.0077,  0.0522, -0.0435,  0.0314],
		[ 0.0013,  0.0495,  0.0320,  0.2127,  0.1564],
		[ 0.0336, -0.1632, -0.0285,  0.0473, -0.0933],
		[ 0.1981,  0.0734, -0.0160,  0.0946,  0.0058]], device='cuda:0',
	       grad_fn=<SelectBackward0>)

	odata[[0,0],[0,1],[0,0]]
	>>tensor([0.2893, 0.2602], device='cuda:0', grad_fn=<IndexBackward0>)

	odata[[0,0,0,0,0,0,0],[0,1,2,3,4,5,6],[1,1,1,1,1,1,1]]
	>>tensor([-0.0434, -0.0337,  0.0756, -0.0077,  0.0495, -0.1632,  0.0734],
	       device='cuda:0', grad_fn=<IndexBackward0>)
        '''

        # train
        eval_raw = self.eval_net(b_s.to(self.device)).reshape(-1,self.num_agents,int(self.num_actions/self.num_agents))
        q_eval = eval_raw.max(2)[0] # shape (batch, 1)

        # conduct mix index for q_next
        a_index = eval_raw.max(2)[1].reshape(-1,self.num_agents) # action from eval_net
        #ii_x = np.array([np.repeat(x,self.num_agents) for x in range(batch_size)]).reshape(-1)
        #ii_y = np.array([range(self.num_agents) for i in range(batch_size)]).reshape(-1)

        #q_next = self.target_net(b_s_.to(self.device)).detach().reshape(-1, self.num_agents, int(self.num_actions / self.num_agents))[ii_x,ii_y,a_index] # detach from graph, don't backpropagate
        q_next = self.target_net(b_s_.to(self.device)).detach().reshape(-1, self.num_agents, int(self.num_actions / self.num_agents))[:,:,a_index] # detach from graph, don't backpropagate

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

    def saveModel(self, path):
        th.save(self.eval_net,path)


