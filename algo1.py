import numpy as np
from random import sample, random
# !pip3 install git+https://github.com/slremy/netsapi --user --upgrade
from netsapi.challenge import *
import itertools
import tensorflow as tf

class version3:

    
    def __init__(self,env):
        tf.reset_default_graph()
        self.n_learning_trials = 100
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.env = env

                
    def train(self):

        best_rew = 0 
  
        
        for i in range(self.n_learning_trials):
            cum_rew = 0
            self.env.reset()
            state = self.env.state
#                 flag = 0
            
            action = [[random.random(), random.random()]]

            nx_state, rew, done, _ = self.env.evaluateAction(action[0])
            if rew > best_rew:
                best_rew = rew
                best_action = action
                
        

#         best_action = [[0,0.78]]
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
    EvaluateChallengeSubmission(ChallengeProveEnvironment, version3, "agent_v1.csv")