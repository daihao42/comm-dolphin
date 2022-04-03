import argparse
import numpy as np
import time, datetime
import pickle
import torch


import numpy as np
import time
from algorithms.dqn import DQN
from algorithms.ddqn import DDQN
from algorithms.dueling_dqn import DuelingDQN
from algorithms.pg import PolicyGradient
from algorithms.maddpg import MADDPG

def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="cn", help="name of the scenario script")
    parser.add_argument("--num-agents", type=int, default=3, help="number of agents in the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=100, help="maximum episode length")
    parser.add_argument("--continuous-actions", type=bool, default=False, help="continuous actions")
    parser.add_argument("--num-episodes", type=int, default=400000, help="number of episodes")
    parser.add_argument("--eval-episodes", type=int, default=1000, help="number of evaluate episodes")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate for Adam optimizer")

    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--random-action", action="store_true", default=False)


    parser.add_argument("--algorithm", type=str, default="dqn", help="the training algorithm")

    parser.add_argument("--num-adversaries", type=int, default=0, help="number of adversaries")
    parser.add_argument("--memory-capacity", type=int, default=1000, help="number of transitions in store memory")
    # Core training parameters
    # Checkpointing
    parser.add_argument("--save-dir", type=str, default="./tmp/policy/", help="directory in which training state and model should be saved")
    parser.add_argument("--load-dir", type=str, default="./tmp/policy/", help="directory in which training state and model are loaded")
    # Evaluation
    return parser.parse_args()


def make_env(arglist):
    import simulators as simulators
    scenario_name = arglist.scenario
    # load scenario from script
    if arglist.scenario == "traffic_junction":
        env = simulators.load(scenario_name + ".py").Scenario(n_max=arglist.num_agents, max_steps=arglist.max_episode_len, display=arglist.display)
    else:
        env = simulators.load(scenario_name + ".py").Scenario(num_agent=arglist.num_agents, max_cycles=arglist.max_episode_len, continuous_actions=arglist.continuous_actions, display=arglist.display)
    return env

import seaborn as sns
import pylab as plt
import pandas as pd

def analysis_parameters(arglist, env, learner):
    lays = len(list(learner.agents[0].policy.named_parameters()))
    agts = len(learner.agents)
    #fig, axs = plt.subplots(agts,lays, sharex=True)
    fig, axs = plt.subplots(agts,lays)
    jjs = 0
    for agent in learner.agents:
        ps = {}
        for name, parameters in agent.policy.named_parameters():
            ts = pd.DataFrame(parameters.reshape(-1,1).cpu().detach().numpy())
            ps[name] = ts
        iis = 0
        for k,v in ps.items():
            sns.histplot(v, ax=axs[jjs][iis],legend=False)
            #sns.scatterplot(x=list(v.index),y=v[0], ax=axs[jjs][iis],legend=False, s=1)
            if jjs == 0:
                axs[jjs][iis].set_title(k)
            axs[jjs][iis].set_ylabel("")
            iis += 1
        jjs += 1

    plt.show()
    
    
def loadModel(arglist, learner):
    path = "saved/{}/{}/{}".format(arglist.scenario, arglist.algorithm, arglist.num_episodes)
    
    if arglist.algorithm == "dqn":
        learner.eval_net.load_state_dict(torch.load(path).state_dict())
    elif arglist.algorithm == "ddqn":
        learner.eval_net.load_state_dict(torch.load(path).state_dict())
    elif arglist.algorithm == "duelingdqn":
        learner.eval_net.load_state_dict(torch.load(path).state_dict())
    elif arglist.algorithm == "maddpg":
        [a.policy.load_state_dict(torch.load(path+"agent_{}".format(i)).state_dict()) for i,a in enumerate(learner.agents)]
    else:
        pass

    return learner

if __name__ == '__main__':
    arglist = parse_args()
    env = make_env(arglist)

    env.reset()

    # global training

    if arglist.scenario == 'traffic_junction':

        observation_shape = env.observation_shape

    else:

        if arglist.algorithm == 'maddpg':
            observation_shape = env.env.observe(env.env.agents[0]).shape

        else:
            observation_shape = env.state().shape
     
    initial_epsilon = 0

    target_replace_iter = 5

    epsilon_decremental = 0

    logger = None

    learnerConstructor = {"maddpg" : MADDPG(env,
                      learning_rate=arglist.lr,
                      initial_epsilon=initial_epsilon,
                      epsilon_decremental=epsilon_decremental,
                      memory_capacity=arglist.memory_capacity, target_replace_iter=target_replace_iter,
                      observation_shape=observation_shape,
                      num_actions=env.action_space,
                      num_agents = env.num_agent,
                      logger = logger)
                   }

    learner = learnerConstructor[arglist.algorithm]

    learner = loadModel(arglist, learner)

    analysis_parameters(arglist, env, learner)
