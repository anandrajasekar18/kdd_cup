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

class version_bayes_v3:

    
    def __init__(self, env):
        self.env = env
        self.env_copy = self.env
        self.envs_copy = []
        self.n_calls = 20
        
        
    
    def get_reward(self,action, env_copy):
        
#         print (env_copy.rewards, env_copy.state)
        new_env_copy = copy.deepcopy(env_copy)
        nx_state, rew, done, _ = new_env_copy.evaluateAction(action)
        
        self.envs_copy.append(new_env_copy)
        return rew

    def train(self):

        best_action = []
        for i in range(self.env_copy.policyDimension):

            r1 = gp_minimize(lambda x: -self.get_reward(x,self.env_copy), 
                        dimensions = [(0.0,1.0)]*2,
                        acq_func='EI',
                        xi=0.1,            
                        n_calls=self.n_calls,         
                        n_random_starts=1 
                       )

            self.env_copy = self.envs_copy[np.argmin(r1.func_vals)]
            self.env_copy.experimentsRemaining-= (self.n_calls-1)

            self.envs_copy = []
            best_action.append(r1.x)
        return best_action

  

    def generate(self):
        
        best_actions = self.train()
        best_policy = {str(state): best_actions[state-1] for state in np.arange(1,self.env_copy.policyDimension+1)}
        best_reward = self.env_copy.evaluatePolicy(best_policy)
#         best_reward = 0
        print(best_policy, best_reward)
        return best_policy, best_reward

if __name__ == "__main__":
    EvaluateChallengeSubmission(ChallengeProveEnvironment, version_bayes_v3, "agent_v3.csv")