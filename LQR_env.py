from gym.core import Env
from gym.spaces.discrete import Discrete
from gym.spaces.box import Box
from control import dlqr
import matplotlib.pyplot as plt
import numpy as np




class LQR_env(Env):
    
    """ LQR environment to test learning continuous action space algorithms- DDPG,TD3,SAC etc."""
    
   
    def __init__(self,env_str='LQR',state_dim=10,action_dim=4,lqr_rew_multiplier=10):
        
        self.action_space=Box(-float('inf'),float('inf'),(action_dim,))
        self.observation_space=Box(-float('inf'),float('inf'),(state_dim,1))
        
        self.env_str=env_str
        self.current_epi=0
        np.random.seed(1234)
        self.state_dim=state_dim
        self.A=np.random.rand(10,10)
        self.B=np.random.rand(10,4)
        sqrt_Q=np.random.rand(10,10)
        sqrt_R=np.random.rand(4,4)*100   
        self.Q=LQR_env.square_matrix(sqrt_Q)
        self.R=LQR_env.square_matrix(sqrt_R)
        ## solve discrete LQR to yeild K- optimal feedback linear controller,
        ## S- optimal state cost function - cost for state X is t(X)*S*X
        ## E- eigenvalues of the close loop Linear dynamic system
        self.K, self.S, self.E = dlqr(self.A, self.B, self.Q, self.R)
        ## max rew/min cost =quadratic_form(S,X_0)==S.sum() for X_0=np.ones
        self.max_lqr_rew=-self.S.sum()
        self.lqr_rew_multiplier=lqr_rew_multiplier
        
        
        
        
        
       
        
    
    
    @staticmethod
    def square_matrix(A:np.array) -> np.array:
        """ square a matrix A: return t(A)*A"""
        return np.matmul(A,np.transpose(A,axes=(1,0)))
    
    @staticmethod
    def quadratic_form(W:np.array,X:np.array) -> float:
        """  return quadratic form  t(X)*W*X"""
        return np.matmul(np.transpose(X,axes=(1,0)),np.matmul(W,X))
    
    def reset(self):
        ## initialize state to ones
        self.X=np.ones((self.state_dim,1))
        self.epi_total_rew=0
        return self.X
    
    def step(self,action):
        action=np.expand_dims(action,1)
        rew=-(LQR_env.quadratic_form(self.Q,self.X)+LQR_env.quadratic_form(self.R,action)).item()
        self.X=np.matmul(self.A,self.X)+np.matmul(self.B,action)
        self.epi_total_rew+=rew
        ### done when reward passes certain threshhold-multiplier*max possible reward or reward doesnt change by 1 pct
        done=self.epi_total_rew<self.lqr_rew_multiplier*self.max_lqr_rew or rew/self.epi_total_rew<0.01
        return self.X,rew,done,{}
    
    
    
def get_LQR_trajectories(X_0,A,B,Q,R,K,S):
    """ get state-avtion trajectories for a LQR controller defined by K
       (the linear optimal feedback controller) and S (optimal quadratic cost fn)
       acting on a plant defined by A,B dynamics and Q,R cost matrices with X_0
       initial state"""
    cost=0
    X=X_0
    t=0
    cost_vs_timesteps=[]
    min_lqr_cost=LQR_env.quadratic_form(S,X)
    while True:
        U=np.matmul(-K,X)
        curr_cost=LQR_env.quadratic_form(Q,X)+LQR_env.quadratic_form(R,U)
        cost+=curr_cost.item()
        cost_vs_timesteps.append(cost)
        if cost/min_lqr_cost>0.99:
            break
        X=np.matmul(A,X)+np.matmul(B,U)
        t+=1
        
    return t,cost_vs_timesteps