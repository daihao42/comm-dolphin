import pylab as plt
from pettingzoo.mpe import simple_spread_v2
import time


## Discrete Action

def dAction():
    env = simple_spread_v2.env(N=3,max_cycles=50)

    env.reset()

    print(env.observe(agent="agent_0"))
    print(env.state())
    for epoch in range(100):
        #print(env.agents)
        for i in env.agents:
            if(not env.dones[env.agent_selection]):
                env.step(env.action_space(agent=env.agent_selection).sample())
            else:
                env.step(None)
            env.render()
            print(epoch, " : ", env.agent_selection,env.rewards)
            time.sleep(0.2)

    env.close()

## Continous Action
def cAction():
    env = simple_spread_v2.env(N=3,max_cycles=50, continuous_actions=True)

    env.reset()

    print(env.observe(agent="agent_0"))
    print(env.state())
    for epoch in range(100):
        #print(env.agents)
        for i in env.agents:
            if(not env.dones[env.agent_selection]):
                env.step(env.action_space(agent=env.agent_selection).sample())
            else:
                env.step(None)
            env.render()
            print(epoch, " : ", env.agent_selection,env.rewards)
            time.sleep(0.2)

    env.close()


if __name__ == '__main__':
    import sys
    if(sys.argv[1] == "d"):
        dAction()
    if(sys.argv[1] == 'c'):
        cAction()
