import optuna
from tqdm import tqdm
import pandas as pd
import numpy as np
from RL_env_utils import *
import gym
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from stable_baselines3 import A2C
from tqdm import tqdm










class trial_objective_tracker():
    def __init__(self,rl_algo,hyperparam_sampler,
                 default_hyperparams,
                 max_timesteps,train_timesteps,eval_episodes,
                 ):
        
        """ class to track agent reward evaluation after every training phase
             agent will learn for a maximum of max_timesteps -(if trial is not pruned).
             Each training phase is train_timesteps long followed by an eval phase
             for eval_apisodes.Objective method is used by optuna to suggest/prune trials"""
        
        self.rl_algo=rl_algo
        self.iter_num=0
        self.default_hyperparams= default_hyperparams
        self.env=default_hyperparams['env']
        self.hyperparam_sampler=hyperparam_sampler
        self.max_timesteps=max_timesteps
        self.train_timesteps=train_timesteps
        self.eval_episodes=eval_episodes
        self.best_trial={}
        self.best_eval_score=-float('inf')
        self.trial_num=0
        
        
        
        
    def objective(self,trial: optuna.Trial) -> float:
        
        self.trial_num+=1
        # setup eval env    
        kwargs = self.default_hyperparams.copy()
        # Sample hyperparameters
        kwargs.update(self.hyperparam_sampler(trial))
        print('starting trial'+str(self.trial_num)+ ' with params: '+str(kwargs))
        # Create the RL model
        model = self.rl_algo(**kwargs)
        nan_encountered = False
        ## num_training phases=self.n_max_timesteps//self.train_timesteps 
        for train_iter in tqdm(range(self.max_timesteps//self.train_timesteps)):
            try:
                self.iter_num+=1
                model.learn(self.train_timesteps)
                eval_df=sample_n_agent_trajectories(self.env,self.eval_episodes,model)
                eval_mean_rew=eval_df['epi_rew'].mean()
                print('reward at trial:'+str(self.trial_num)+' iteration:'+str(train_iter)+' = '+str(eval_mean_rew))
                trial.report(eval_mean_rew, train_iter)
                if trial.should_prune():
                    print('trial_pruned')
                    raise optuna.TrialPruned()
                
                ## save best (highest scoring model across all trials and iterations)
                if self.best_eval_score<eval_mean_rew:
                    print('new best model found with kwargs: '+str(kwargs)+' at iteration: '+str(train_iter))
                    self.best_eval_score=eval_mean_rew
                    self.best_model=model
                    self.best_hyperparams=kwargs
            except AssertionError as e:
                print(e)
                nan_encountered = True
                return float("nan")
            #finally:
                # Free memory
             #   model.env.close()
              #  eval_env.close()

        return  eval_mean_rew









def run_RL_hyperparameter_opt(env_cls=custom_pong_env,n_trials=100,
                              max_timesteps=20000,
                              train_timesteps=100,
                              eval_episodes=3,
                             algo_string='A2C'):
    
    DEFAULT_HYPERPARAMS = {
    "policy": "MlpPolicy",
    "env": env_cls(),
    }
    
    ## TODO
    RL_ALGO_DICT={'A2C':A2C
                 }
    
    
    ## 10% of trails before starting to prune
    N_STARTUP_TRIALS=int(0.05*n_trials)
    ## max budget for each trial == max_timesteps
    N_EVALUATIONS =max_timesteps//train_timesteps
    # Set pytorch num threads to 1 for faster training
    torch.set_num_threads(1)

    sampler = TPESampler(n_startup_trials=N_STARTUP_TRIALS)
    # Do not prune before 1/3 of the max budget is used
    pruner = MedianPruner(n_startup_trials=N_STARTUP_TRIALS, n_warmup_steps=N_EVALUATIONS // 3)

    study = optuna.create_study(sampler=sampler, pruner=pruner, direction="maximize")
    try:
        tracker=trial_objective_tracker(rl_algo=RL_ALGO_DICT[algo_string],hyperparam_sampler=sample_a2c_params,
                                        default_hyperparams=DEFAULT_HYPERPARAMS,max_timesteps=max_timesteps,
                                        train_timesteps=train_timesteps,eval_episodes=eval_episodes)
                                            
        
        study.optimize(tracker.objective, n_trials=n_trials, timeout=1000)
    except KeyboardInterrupt:
        pass

    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    print("  User attrs:")
    for key, value in trial.user_attrs.items():
        print("    {}: {}".format(key, value))
        
    return tracker



        

def sample_a2c_params(trial: optuna.Trial) -> Dict[str, Any]:
    """Sampler for A2C hyperparameters."""
    #gamma = 1.0 - trial.suggest_float("gamma", 0.0001, 0.1, log=True)
    max_grad_norm = trial.suggest_float("max_grad_norm", 0.3, 5.0, log=True)
    gae_lambda = 1.0 - trial.suggest_float("gae_lambda", 0.001, 0.2, log=True)
    ## n_stepsk== env steps before learning gradient update
    n_steps = 2 ** trial.suggest_int("exponent_n_steps", 3, 5)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1, log=True)
    ent_coef = trial.suggest_float("ent_coef", 0.00000001, 0.1, log=True)
    ortho_init = trial.suggest_categorical("ortho_init", [False, True])
    #net_arch = trial.suggest_categorical("net_arch", ["tiny", "small"])
    activation_fn = trial.suggest_categorical("activation_fn", ["tanh", "relu"])
    n_layers =  trial.suggest_int("n_layers",1,2)
    layer_size=trial.suggest_categorical("layer_size", [32,64,128,256])

    # Display true values
   # trial.set_user_attr("gamma_", gamma)
    trial.set_user_attr("gae_lambda_", gae_lambda)
    trial.set_user_attr("n_steps", n_steps)
    
    layers=[layer_size for i in range(n_layers)]
    net_arch = [
        {"pi": layers, "vf": layers}
    ]
    
   # net_arch = [
    #    {"pi": [64], "vf": [64]} if net_arch == "tiny" else {"pi": [64, 64], "vf": [64, 64]}
    #]


    activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU}[activation_fn]

    return {
        "n_steps": n_steps,
        #"gamma": gamma,
        "gae_lambda": gae_lambda,
        "learning_rate": learning_rate,
        "ent_coef": ent_coef,
        "max_grad_norm": max_grad_norm,
        "policy_kwargs": {
            "net_arch": net_arch,
            "activation_fn": activation_fn,
            "ortho_init": ortho_init,
        
        },
    }

        
        
        
if __name__=='__main__':
    best_score=run_RL_hyperparameter_opt(max_timesteps=2000)