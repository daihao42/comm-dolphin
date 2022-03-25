#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gym,time

env = gym.make('ma_gym:TrafficJunction10-v0')
done_n = [False for _ in range(env.n_agents)]
ep_reward = 0
obs_n = env.reset()

while not all(done_n):
    env.render()
    obs_n, reward_n, done_n, info = env.step(env.action_space.sample())
    print(reward_n)
    ep_reward += sum(reward_n)
    print(ep_reward)
    time.sleep(1)
    #env.close()

env.close()

#if __name__ == '__main__':

