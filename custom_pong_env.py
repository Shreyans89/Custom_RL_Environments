import numpy as np
import torch
import gym
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from gym.core import Env
from gym import spaces
import pandas as pd
from gym.spaces.discrete import Discrete
from gym.spaces.box import Box
from datetime import datetime

def makenow(foldername):
    """convert foldername to foldername_datetime_now"""
    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
    out='./'+foldername+'_'+dt_string+'/'
    return out



class custom_pong_env(Env):
    
    """ Wrapper Class to wrap around actual env and 
    return processed next state,done and reward info
    so Qnet can ingest processed environment state to produce
    actions. Example:
    
    In Atari games the Kframes argument sets the number of frames
    to be used by the agent to produce an action.the agent repeats
    the action chosen for kframes=4 consecutive atari frames
    and concats the 4 produced observation into one tensor(frames) of
    shape 4 by 84 by 84 -4 greyscale images of size 84 by 84.
    
    The reset() method resets the wrapped atari env and also executes
    a random action kframes=4 times to produce the first video"""
    
   
    def __init__(self,env_str='Pong-v0',gamma=0.95,kframes=4,valid_actions=[2,3],num_features=6,noop=0,
                 saveframes=False,frames_dir='saved_frames',
                obj2RGB={'background':torch.tensor([0.5647, 0.2824, 0.0667]),
           'opp':torch.tensor([0.8353, 0.5098, 0.2902]),
           'puck':torch.tensor([0.9255, 0.9255, 0.9255]),
           'plyr':torch.tensor([0.3608, 0.7294, 0.3608])}):
        
        self.action_space=Discrete(len(valid_actions))
        self.observation_space=Box(-1,1,[num_features])
        
        self.env_str=env_str
        self.env=gym.make(env_str)
        self.kframes=kframes
        self.valid_actions=valid_actions ## actions are 0:noop,2:up,3:down
        self.gamma=gamma
        self.obj2RGB=obj2RGB
        self.noop=noop
        self.current_epi=0
        self.saveframes=saveframes
        if self.saveframes:
            self.frames_dir=Path(frames_dir)
            self.frames_dir.mkdir(exist_ok=True)
            
    ## move to comply with rllibs stuff
    def _setup_spaces(self):
        self.action_space=Discrete(len(valid_actions))
        self.observation_space=Box(-1,1,[num_features])
        
        
    def get_next_kframes(self,action):
        """ return the next k frames from the atari sim while performing the same action"""
        frames=[]
        for i in range(self.kframes):
            frame,_,_,_=self.env.step(action)
            frames.append(frame)
        frames=np.stack(frames)
        return frames
        
    def preproc(self,frames):
        """pre process the atari frames and convert them to object (puck/player/opponent) masks"""
        proc_frames=torch.tensor(np.transpose(frames,(0,3,1,2))/255.)
        proc_frames=TF.crop(proc_frames,top=40,left=0,height=150,width=160)
        cols=torch.stack(list(self.obj2RGB.values()))
        obj_masks=((cols.unsqueeze(2).unsqueeze(3).unsqueeze(0)-proc_frames.unsqueeze(1))**2).sum(dim=2).argmin(dim=1)
        # remove frames where object:plyr,puck,opponent are not in view
        obj2masks={obj:obj_masks==i for i,obj in enumerate(self.obj2RGB) if not (obj_masks==i).sum()==0}
        return obj2masks,proc_frames.mean(dim=0)
    
    
 
    
    def extract_features(self,obj2masks,mean_frame):
        """ extract object positions and velocities from their pixel masks"""
        obj2coords={obj:custom_pong_env.obj_x_y_from_masks(masks[masks.sum(dim=(1,2))>0]) for obj,masks in obj2masks.items()}
        for obj in self.obj2RGB:
            if obj not in obj2coords:
                obj2coords[obj]=self.last_known_locs[obj]## object is out of sight, use last known location
        self.last_known_locs=obj2coords ## update last known locations and velocities of object
        
        plyr_x,opp_x=obj2coords['plyr']['x'],obj2coords['opp']['x']
        frame_H,frame_W=mean_frame.shape[1],plyr_x-opp_x
        
        
        feature_vector={'opp_y':1-obj2coords['opp']['y']/frame_H,
                         'puck_x':(obj2coords['puck']['x']-opp_x)/frame_W,
                        'puck_y':1-obj2coords['puck']['y']/frame_H,
                        'puck_vx':obj2coords['puck']['vx']/frame_W,
                        'puck_vy':-obj2coords['puck']['vy']/frame_H,
                       'plyr_y':1-obj2coords['plyr']['y']/frame_H
                       }
        
        if self.saveframes:
            self.save_frames(feature_vector,mean_frame)
        return feature_vector
    
   
    def save_frames(self,feature_vector,frame):
        feature_vector['selected_action']=self.current_action
        feature_vector['current_episode']=self.current_epi
        fig, ax = plt.subplots()
        frame=torch.permute(frame,(1,2,0))
        ax.imshow(frame)
        ax.text(frame.shape[0]/2, frame.shape[1]/2, str(feature_vector), fontsize = 5)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        filename='episode_'+str(self.current_epi)+'_' + 'frame_' + str(self.current_frame)+'.png'
        fig.savefig(self.frames_dir/filename, bbox_inches='tight', dpi=1000)
        fig.close()
        
        
    @staticmethod
    def obj_x_y_from_masks(masks):
        ones=torch.ones_like(masks)
        pos_x=(torch.cumsum(ones,dim=2)*masks).sum(dim=(1,2))/masks.sum(dim=(1,2))
        pos_y=(torch.cumsum(ones,dim=1)*masks).sum(dim=(1,2))/masks.sum(dim=(1,2))
        ## calculate position and velocity of objects
        return {'x':torch.round(pos_x.mean()),'y':torch.round(pos_y.mean()),
                'vx':pos_x[-1]-pos_x[0],'vy':pos_y[-1]-pos_y[0]}
    
    def __str__(self):
        return 'custom_'+self.env_str
        
    
    
    def step(self,action):
        ## to map 0/1 actions to appropriate atari controls
        action=self.valid_actions[action]
        ## save current action to be plotted/outputted
        self.current_action=action
        self.current_frame+=1
        frames=self.get_next_kframes(action)
        obs=self.extract_features(*self.preproc(frames))
        plyr_won,opp_won=obs['puck_x']<0,obs['puck_x']>1
        rew=(1*plyr_won-1*opp_won).item() ## when either the puck goes beyond the player or the opponent
        ## discounting rew in the environment
        rew*=self.gamma**self.current_frame
        done=(plyr_won or opp_won).item()
        return np.array(list(obs.values())),rew,done,{}
        
    
    def reset(self):
        self.env.reset()
        self.current_epi+=1
        self.current_frame=0
        onscreen_objs={}
    ## puck disappears for several frames after reset.do nothing till puck appears-noop
        i=0
        while not 'puck' in onscreen_objs:
            i+=1
            #print('waiting for '+str(i)+'steps')
            onscreen_objs,_=self.preproc(self.get_next_kframes(self.noop))
        return self.step(self.noop)[0]
    
    



        

        
