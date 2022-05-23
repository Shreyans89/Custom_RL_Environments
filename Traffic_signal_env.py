import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from gym.core import Env
import gym.spaces as spaces 
from scipy.integrate import solve_ivp





def sawtooth_flux(rho,rho_min=0.,rho_mid=0.5,rho_max=1.,flux_max=0.1*1,flux_min=0):
    return np.minimum( (rho-rho_min)/(rho_mid-rho_min)*flux_max,(rho_max-rho_min)/(rho_max-rho_mid)*flux_max*(rho_max-rho)/(rho_max-rho_min))




def cell_interface_flux(t,y,rightmost_rho=np.array([0.,1.]).reshape(2,1),leftmost_rho=np.zeros((2,1)),rho_mid=0.5,flux_fn=sawtooth_flux):
    
    rho_vec=(y.reshape(2,len(y)//2))[:,:-1]
    
    rho_left=np.concatenate((leftmost_rho,rho_vec),axis=1)
    rho_right=np.concatenate((rho_vec,rightmost_rho),axis=1)
    fluxes=flux_fn(np.concatenate((leftmost_rho,rho_vec,rightmost_rho),axis=1))
    fluxes_left=fluxes[:,:-1]
    fluxes_right=fluxes[:,1:]
    fluxes_min=np.minimum(fluxes_left,fluxes_right)
    fluxes_max=flux_fn(rho_mid*np.ones_like(fluxes_min))

    
    conds=np.stack(arrays=[((rho_left<=rho_mid) * (rho_right<=rho_mid)),
                           ((rho_left>=rho_mid) * (rho_right>=rho_mid)),
                           ((rho_left<=rho_mid) * (rho_right>=rho_mid)),
                            ((rho_left>=rho_mid) * (rho_right<=rho_mid))])
   
    gudunov_fluxes=(np.stack([fluxes_left,fluxes_right,fluxes_min,fluxes_max])*conds).sum(axis=0)
    ## assuming left is in and right is out
    d_rho_dt=gudunov_fluxes[:,:-1]-gudunov_fluxes[:,1:]
    ## .unsqueeze() to meet size
    
    ## L2 Penalty
    penalty_growth_rate=np.expand_dims((rho_vec**2).sum(axis=1),axis=1)
    
    #L1 penalty
   ##penalty_growth_rate=np.expand_dims((np.abs(rho_vec)).sum(axis=1),axis=1)

    return np.concatenate((d_rho_dt,penalty_growth_rate),axis=1).flatten()

   # flatten flux for both intersection as solve IVP expects a 1 d array 
    
    
class TrafficSignal(Env):
    def __init__(self,min_time,max_time,thresh=0.1,n_segments=10,rho_min=0.,rho_max=[1.,1.]):
        super(TrafficSignal,self).__init__()
        # Action space, min/max switch time min=5,max=10
        self.action_space = spaces.Box(low=min_time,high=max_time,shape=(1,))
        # Temperature array
        self.observation_space = spaces.Box(low=rho_min, high=max(rho_max),shape=(2,n_segments))
        # Set start temp
        self.rho_max=rho_max
        self.n_segments=n_segments
        self.thresh=thresh
        self.reset()
     
        
        
    def step(self, action):
        sol= solve_ivp(fun=cell_interface_flux,
                       t_span=(0,action.item()),
                       y0=np.concatenate((self.state,np.zeros((2,1))),axis=1).flatten(),
                       t_eval=np.arange(0,action.item(),1.),vectorized=False)
        light_colour=self.num_switches%2
        self.light_colour.append(light_colour)

        if bool(light_colour):
            self.buffer_x.append(sol.y[(self.n_segments+1):,:])
            self.buffer_y.append(sol.y[:-(self.n_segments+1),:])
        else:
            self.buffer_y.append(sol.y[(self.n_segments+1):,:])
            self.buffer_x.append(sol.y[:-(self.n_segments+1),:])
            
            
            
        next_state=sol.y[:,-1].reshape(2,self.n_segments+1)
                                     
       
        self.state =np.matmul(np.array([0,1,1,0]).reshape(2,2),next_state[:,:-1])
                           
        reward=-(next_state[:,-1]).sum()
      
       
        if self.state.max() <= self.thresh: 
           self.done=True
        else:
            self.done=False
        
        self.num_switches+=1
        
        info = {}
        
        # Return step information
        return self.state, reward, self.done, info

    def render(self):
        # Implement viz
        pass
    
    def reset(self):
        self.state =np.array(self.rho_max).reshape(2,1)* np.random.rand(2,self.n_segments)
        self.buffer_x,self.buffer_y,self.light_colour=[],[],[]
        self.num_switches=0
        if self.state.max() <= self.thresh: 
            self.done=True
        else:
            self.done=False
        return self.state