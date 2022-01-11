# import torch
# flag = torch.cuda.is_available()
# print(flag)

# ngpu= 1
# Decide which device we want to run on
# device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
# print(device)
# print(torch.cuda.get_device_name(0))
# print(torch.rand(3,3).cuda()) 

import argparse
import numpy as np
import time, datetime
import pickle


import numpy as np
import time
from algorithms.dqn import DQN
from algorithms.ddqn import DDQN
from algorithms.dueling_dqn import DuelingDQN


def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="cn", help="name of the scenario script")
    parser.add_argument("--num-agents", type=int, default=3, help="number of agents in the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=40, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=400000, help="number of episodes")
    parser.add_argument("--continuous-actions", type=bool, default=False, help="continuous actions")

    parser.add_argument("--display", action="store_true", default=False)

    parser.add_argument("--algorithm", type=str, default="dqn", help="the training algorithm")

    parser.add_argument("--log-dir", type=str, default="dqn", help="logs dir")

    parser.add_argument("--num-adversaries", type=int, default=0, help="number of adversaries")
    parser.add_argument("--good-policy", type=str, default="maddpg", help="policy for good agents")
    parser.add_argument("--adv-policy", type=str, default="maddpg", help="policy of adversaries")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=128, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=128, help="number of units in the mlp")
    parser.add_argument("--prior-batch-size", type=int, default=2000, help="number of samples to optimize at the same time for prior network")
    parser.add_argument("--prior-buffer-size", type=int, default=400000, help="prior network training buffer size")
    parser.add_argument("--prior-num-iter", type=int, default=10000, help="prior network training iterations")
    parser.add_argument("--prior-training-rate", type=int, default=20000, help="prior network training rate")
    parser.add_argument("--prior-training-percentile", type=int, default=80, help="control threshold for KL value to get labels")
    # Checkpointing
    parser.add_argument("--exp-name", type=str, default='exp', help="name of the experiment")
    parser.add_argument("--save-dir", type=str, default="./tmp/policy/", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=1000, help="save model once every time this many episodes are completed")
    parser.add_argument("--load-dir", type=str, default="./tmp/policy/", help="directory in which training state and model are loaded")
    # Evaluation
    parser.add_argument("--restore_all", action="store_true", default=False)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--benchmark-iters", type=int, default=100000, help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default="./benchmark_files/", help="directory where benchmark data is saved")
    parser.add_argument("--plots-dir", type=str, default="./learning_curves/", help="directory where plot data is saved")
    return parser.parse_args()


def make_env(arglist):
    import simulators as simulators
    scenario_name = arglist.scenario
    # load scenario from script
    env = simulators.load(scenario_name + ".py").Scenario(num_agent=arglist.num_agents, max_cycles=arglist.max_episode_len, continuous_actions=arglist.continuous_actions, display=arglist.display)
    return env

from torch.utils.tensorboard import SummaryWriter       

def Logger(ldir):
    writer = SummaryWriter("logs/{}_log_{}".format(ldir,datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")))
    return writer

def global_train(arglist, env, learner):

    obs_n = env.state()
        
    rollouts = 0

    for epoch in range(arglist.num_episodes):

        g_action_n = learner.choose_action(obs_n)

        r_action_n = np.array(g_action_n).reshape(-1, 5)

        action_n = [np.argmax(x) for x in r_action_n]

        new_obs_n, rew_n, done_n, info_n = env.step(action_n)

        new_obs_n = env.state()

        # global reward
        learner.store_transition(np.array(obs_n), np.array(action_n), rew_n["agent_0"], np.array(new_obs_n))

        #logger.add_scalar('Global/Reward\\', rew_n["agent_0"], epoch)

        #print(epoch, rew_n, action_n, done_n, info_n)

        env.render()

        #done = all(done_n)
        done = any(done_n)

        obs_n = new_obs_n

        #time.sleep(0.1)
        
        rollouts += 1

        epoch += 1

        logger.add_scalar('Global/Reward\\', rew_n["agent_0"], epoch)

        if(rollouts % arglist.max_episode_len == 0):
            logger.add_scalar('Global/Final_Reward\\', rew_n["agent_0"], epoch)

        if(rollouts % learn_step == 0):
            print("-------------- start training -------------")
            learner.learn(gamma=arglist.gamma, batch_size=arglist.batch_size)
            print("-------------- end training -------------")

        if done:
            env.close()
            env.reset()
            obs_n = env.state()
        else:
            pass

    learner.saveModel("saved/{}-{}".format(arglist.algorithm, arglist.num_episodes))



if __name__ == '__main__':
    arglist = parse_args()
    env = make_env(arglist)

    env.reset()

    # global training

    obs_n = env.state()
     
    learn_step = arglist.batch_size

    initial_epsilon = 0.5

    target_replace_iter = 5

    epsilon_decremental = (1-initial_epsilon) / (arglist.num_episodes / learn_step / target_replace_iter)

    logger = Logger(arglist.log_dir)
   
    def conductLearner(algorithm):
        learner = {"dqn" : DQN(env,
                      initial_epsilon=initial_epsilon,
                      epsilon_decremental=epsilon_decremental,
                      memory_capacity=5000, target_replace_iter=target_replace_iter,
                      learning_rate=arglist.lr,
                      observation_shape=obs_n.shape,
                      num_actions=env.action_space * env.num_agent,
                      num_agents = env.num_agent,
                      logger = logger),
                    "ddqn" : DDQN(env,
                      initial_epsilon=initial_epsilon,
                      epsilon_decremental=epsilon_decremental,
                      memory_capacity=5000, target_replace_iter=target_replace_iter,
                      learning_rate=arglist.lr,
                      observation_shape=obs_n.shape,
                      num_actions=env.action_space * env.num_agent,
                      num_agents = env.num_agent,
                      logger = logger),
                     "duelingdqn" : DuelingDQN(env,
                      initial_epsilon=initial_epsilon,
                      epsilon_decremental=epsilon_decremental,
                      memory_capacity=5000, target_replace_iter=target_replace_iter,
                      learning_rate=arglist.lr,
                      observation_shape=obs_n.shape,
                      num_actions=env.action_space * env.num_agent,
                      num_agents = env.num_agent,
                      logger = logger)
                   }
        return learner[algorithm]


    learner = conductLearner(arglist.algorithm)

    global_train(arglist, env, learner)

