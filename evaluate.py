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
    env = simulators.load(scenario_name + ".py").Scenario(num_agent=arglist.num_agents, max_cycles=arglist.max_episode_len, continuous_actions=arglist.continuous_actions, display=arglist.display)
    return env

def global_evaluate(arglist, env, learner):

    obs_n = env.state()
        
    for epoch in range(arglist.eval_episodes):

        g_action_n = learner.choose_action(obs_n)

        r_action_n = np.array(g_action_n).reshape(-1, 5)

        action_n = [np.argmax(x) for x in r_action_n]

        new_obs_n, rew_n, done_n, info_n = env.step(action_n)

        new_obs_n = env.state()

        env.render()

        #done = all(done_n)
        done = any(done_n)

        obs_n = new_obs_n

        time.sleep(0.1)

        if done:
            env.close()
            env.reset()
            obs_n = env.state()
        else:
            pass


def global_policy_evaluate(arglist, env, learner):

    obs_n = env.state()

    done_n = [False for x in range(env.num_agent)]

    for epoch in range(arglist.eval_episodes):

        action_n = learner.choose_action(obs_n)

        new_obs_n, rew_n, done_n, info_n = env.step(action_n)

        new_obs_n = env.state()

        env.render()

        #done = all(done_n)
        done = any(done_n)

        obs_n = new_obs_n

        time.sleep(0.1)

        if done:
            env.close()
            env.reset()
            obs_n = env.state()
        else:
            pass



def maddpg_evaluate(arglist, env, learner):

    obs_n = [env.env.observe(i) for i in env.env.agents]
        
    done_n = [False for x in range(env.num_agent)]

    for epoch in range(arglist.eval_episodes):

        g_action_n = learner.choose_action(obs_n)

        action_n = [np.argmax(x) for x in g_action_n]

        action_n, g_action_n = markDone(done_n, action_n, g_action_n)

        new_obs_n, rew_n, done_n, info_n = env.step(action_n)

        env.render()

        obs_n = new_obs_n

        time.sleep(0.1)

        done = all(done_n)
        #done = any(done_n)

        if done:
            env.close()
            env.reset()
            obs_n = [env.env.observe(i) for i in env.env.agents]
            done_n = [False for x in range(env.num_agent)]
        else:
            pass


def markDone(done_n, action_n, g_action_n):
    '''
    since the action of a done agent must be 'None', marked it as 0, which means no_action in env.
    '''
    maction_n = []
    saved_action_n = []
    for i in range(len(action_n)):
        if done_n[i]:
            maction_n.append(None)
            saved_action_n.append(np.zeros(len(g_action_n[i])))
        else:
            maction_n.append(action_n[i])
            saved_action_n.append(g_action_n[i])
    return maction_n, saved_action_n


def loadModel(arglist, learner):
    path = "saved/{}-{}-{}".format(arglist.scenario, arglist.algorithm, arglist.num_episodes)
    
    loader = {"dqn": learner.eval_net.load_state_dict(torch.load(path)),
            "ddqn": learner.eval_net.load_state_dict(torch.load(path)),
              "duelingdqn": learner.eval_net.load_state_dict(torch.load(path)),
              "maddpg": [a.policy.load_state_dict(torch.load(path+"agent_{}".format(i))) for i,a in enumerate(learner.agents)]
    }
    loader[arglist.algorithm]
    return learner

if __name__ == '__main__':
    arglist = parse_args()
    env = make_env(arglist)

    env.reset()

    # global training

    obs_n = env.state()
     
    initial_epsilon = 0

    target_replace_iter = 5

    epsilon_decremental = 0

    logger = None

    learnerConstructor = {"dqn" : (DQN(env,
                      initial_epsilon=initial_epsilon,
                      epsilon_decremental=epsilon_decremental,
                      memory_capacity=arglist.memory_capacity, target_replace_iter=target_replace_iter,
                      learning_rate=arglist.lr,
                      observation_shape=obs_n.shape,
                      num_actions=env.action_space * env.num_agent,
                      num_agents = env.num_agent,
                      logger = logger), global_evaluate),

                      "ddqn" : (DDQN(env,
                      initial_epsilon=initial_epsilon,
                      epsilon_decremental=epsilon_decremental,
                      memory_capacity=arglist.memory_capacity, target_replace_iter=target_replace_iter,
                      learning_rate=arglist.lr,
                      observation_shape=obs_n.shape,
                      num_actions=env.action_space * env.num_agent,
                      num_agents = env.num_agent,
                      logger = logger), global_evaluate),

                      "duelingdqn" : (DuelingDQN(env,
                      initial_epsilon=initial_epsilon,
                      epsilon_decremental=epsilon_decremental,
                      memory_capacity=arglist.memory_capacity, target_replace_iter=target_replace_iter,
                      learning_rate=arglist.lr,
                      observation_shape=obs_n.shape,
                      num_actions=env.action_space * env.num_agent,
                      num_agents = env.num_agent,
                      logger = logger), global_evaluate),

                      "policygradient" : (PolicyGradient(env,
                      learning_rate=arglist.lr,
                      observation_shape=obs_n.shape,
                      num_actions=env.action_space * env.num_agent,
                      num_agents = env.num_agent,
                      logger = logger), global_policy_evaluate),

                      "maddpg" : (MADDPG(env,
                      learning_rate=arglist.lr,
                      initial_epsilon=initial_epsilon,
                      epsilon_decremental=epsilon_decremental,
                      memory_capacity=arglist.memory_capacity, target_replace_iter=target_replace_iter,
                      observation_shape=env.env.observe(env.env.agents[0]).shape,
                      num_actions=env.action_space,
                      num_agents = env.num_agent,
                      logger = logger), maddpg_evaluate)
                   }

    learner, train_func = learnerConstructor[arglist.algorithm]

    learner = loadModel(arglist, learner)

    train_func(arglist, env, learner)
