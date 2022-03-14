#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import deque
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

import random

from torch.autograd import Variable

from models.ActorNet import ActorNet

__author__ = 'dai'

class PolicyGradient():

    # no more epsilon
    def __init__(self, env, learning_rate, observation_shape, num_actions, num_agents, logger) -> None:

        self.num_actions = num_actions

        self.num_agents = num_agents

        self.logger = logger

        self.train_step = 0

        self.prob_net = ActorNet(observation_shape, num_actions)

        # GPU training
        self.loss = []
        self.device = th.device('cuda' if th.cuda.is_available() else 'cpu')
        self.prob_net.to(self.device)
        self.optimizer = th.optim.Adam(self.prob_net.parameters(), lr=learning_rate)

        self.clear_transition()

    def store_transition(self, s, a, r):
        self.ep_observations.append(s)
        self.ep_actions.append(a)
        self.ep_rewards.append(r)

    def clear_transition(self):
        self.ep_observations = []
        self.ep_actions = []
        self.ep_rewards = []

    def choose_action(self,x, greedy=False):
        x = th.tensor(x, dtype = th.float).to(self.device)
        _logits = self.prob_net(x).reshape(self.num_agents, -1)
        _probs = th.softmax(_logits, dim=1)

        '''
        ## debug
        if(th.isnan(self.prob_net.fc1.weight).any() or th.isnan(_probs).any() or th.isinf(_probs).any()):
            import ipdb
            ipdb.set_trace()
        '''

        if(greedy):
            return th.argmax(_probs,dim=1,keepdim=True).cpu().detach().numpy()
        return th.multinomial(_probs, num_samples=1).reshape(-1).cpu().detach().numpy()

    def _discount_and_norm_rewards(self, gamma):
        discounted_ep_rs = np.zeros_like(self.ep_rewards)
        running_add = 0
        for t in reversed(range(0,len(self.ep_rewards))):
            running_add = running_add * gamma + self.ep_rewards[t]
            discounted_ep_rs[t] = running_add
        discounted_ep_rs -= np.mean(discounted_ep_rs)
        discounted_ep_rs /= np.std(discounted_ep_rs)
        return list(map(lambda x:np.repeat(x,self.num_agents),discounted_ep_rs))

    def learn(self, gamma):

        # prediction
        _logits = self.prob_net(th.FloatTensor(self.ep_observations).to(self.device)).reshape(-1, self.num_agents, int(self.num_actions/self.num_agents))

        # cross-entropy basicly include the one-hot encode function, tensor([[0.1,0.5,0.4],[0.2,0.6,0.2]]) v.s. tensor([1,1])

        prediction = Categorical(F.softmax(_logits))

        ploss = -prediction.log_prob(th.LongTensor(self.ep_actions).to(self.device)) # "-" because it was built to work with gradient descent, but we are using gradient ascent

        discounted_ep_rs = self._discount_and_norm_rewards(gamma=gamma)

        pseudo_loss = th.sum(ploss * th.FloatTensor(discounted_ep_rs).to(self.device))

        # back propagation
        self.optimizer.zero_grad()
        pseudo_loss.backward()
        self.optimizer.step()

        ## debug
        if(th.isnan(self.prob_net.fc1.weight).any()):
            import ipdb
            ipdb.set_trace()


        self.logger.add_scalar('Global/Loss\\', pseudo_loss.detach().cpu().numpy(),self.train_step)

        self.train_step += 1

    def saveModel(self, path):
        th.save(self.prob_net,path)

