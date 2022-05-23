import pandas as pd
import numpy as np

def sample_agent_trajectories(env,agent=None):
    """returns a tuple containing list of obs and rews from start to the end of the episopde"""
    ## grab the start ob
    ob=env.reset()
    done=False
    obs,epi_rew,epi_len=[ob],0,0
    # discount factor
   
    actions=[]
    while not done:
        if agent is None :
            action=env.action_space.sample()
        else:
            action,_=agent.predict(ob,deterministic=True)
      
        ob,rew,done,_=env.step(action)
       
        obs.append(ob)
        actions.append(action)
        epi_rew+=rew
        epi_len+=1
    return {'obs':np.stack(obs),'epi_rew':epi_rew,'epi_last_ob':ob,'action':np.stack(actions),'epi_len':epi_len}

def sample_n_agent_trajectories(env,n,agent=None):
    return pd.DataFrame([sample_agent_trajectories(env,agent) for i in range(n)])


class high_low_agent():
    """ an agent/policy which always returns hi/low from an envs action space.
        use to evaluate range of rewards"""
    
    def __init__(self,env,high_low='high'):
        self.action=env.action_space.high if high_low=='high' else env.action_space.low
    
    def predict(self,ob,deterministic=True):
        return self.action,{}
    