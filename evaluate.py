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
import time
import pickle
from models.ActorNet import ActorNet


def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="cn", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=40, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=400000, help="number of episodes")
    parser.add_argument("--num-adversaries", type=int, default=0, help="number of adversaries")
    parser.add_argument("--good-policy", type=str, default="maddpg", help="policy for good agents")
    parser.add_argument("--adv-policy", type=str, default="maddpg", help="policy of adversaries")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=800, help="number of episodes to optimize at the same time")
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
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--benchmark-iters", type=int, default=100000, help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default="./benchmark_files/", help="directory where benchmark data is saved")
    parser.add_argument("--plots-dir", type=str, default="./learning_curves/", help="directory where plot data is saved")
    return parser.parse_args()


def make_env(arglist):
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios
    scenario_name = arglist.scenario
    benchmark = arglist.benchmark
    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env


if __name__ == '__main__':
    arglist = parse_args()
    env = make_env(arglist)
    obs_n = env.reset()
    actions = env.action_space

    import numpy as np
    import time
    from algorithms.dqn import DQN

    def createLearners():
        learners = []
        for i in range(len(env.agents)):
            net = ActorNet(observation_shape=obs_n[0].shape,num_actions=len(actions))
            learner = DQN(env,0.001,1000,100,0.001,net)
            learners.append(learner)
        return learners


    learners = createLearners()

    print(learners)

    rollouts = 0

    while True or not done: 

        action_n = [[learners[i].choose_action([obs_n[i]]).detach().numpy()[0]] for i in range(len(learners))]

        time.sleep(0.1)

        new_obs_n, rew_n, done_n, info_n = env.step(action_n)

        for i in range(len(learners)):
            learners[i].store_transition(obs_n[i], action_n[i],rew_n[i],new_obs_n[i])

        print(rew_n, action_n, done_n, info_n)

        env.render()

        done = all(done_n)

        obs_n = new_obs_n
        
        rollouts += 1

        if(rollouts == 50):
            print("-------------- start training -------------")
            for learn in learners:
                learn.learn()
            rollouts = 0
            print("-------------- end training -------------")

        if done:
            break
        else:
            pass
