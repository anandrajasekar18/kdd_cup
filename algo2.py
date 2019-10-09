import numpy as np
from random import sample, random
import itertools

from netsapi.challenge import *
from skopt import gp_minimize
from skopt.learning import GaussianProcessRegressor
from skopt.learning.gaussian_process.kernels import ConstantKernel, Matern
import copy
import warnings
warnings.filterwarnings("ignore")

class version_bayes:

    
    def __init__(self, env):
        self.env = env
    
    def get_reward(self, action):

        self.env.reset()
        state = self.env.state
        nx_state, rew, done, _ = self.env.evaluateAction(action)   
        
        return rew
  
                
    def train(self):


        r = gp_minimize(lambda x: -self.get_reward(x), 
                [(0.0,1.0),(0.0,1.0)],
                acq_func='EI',     
                xi=0.1,            
                n_calls=100,         
                n_random_starts=5, 
                n_jobs = -1  
               )
        best_action = [r.x] 
        for i in range(1,5):
            best_action.append(best_action[i-1][::-1])
        
        return best_action
    def generate(self):
        
        best_actions = self.train()
        self.env.reset()
        best_policy = {str(state): best_actions[state-1] for state in np.arange(1,self.env.policyDimension+1)}
        best_reward = self.env.evaluatePolicy(best_policy)
#         best_reward = 0
        print(best_policy, best_reward)
        return best_policy, best_reward

if __name__ == "__main__":
    EvaluateChallengeSubmission(ChallengeProveEnvironment, version3, "agent_v3.csv")