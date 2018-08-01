import random, numpy as np, gym
from gym import wrappers
env = gym.make('MountainCar-v0')


Iterations = 6000

best_policy = None 



def run(iteration, best_reward, best_policy, environment):
    max_steps = 200
    reward = 0
    steps = 0
    policy = (best_policy + np.random.rand(2)) - 0.5
    observation = env.reset()
    running = True
    obs_list = []
    print("Test Policy: " + str(policy))
    while running:
        #env.render()
        steps += 1
        if policy.dot(observation) <= 0:
            action = 0
                  
        else:
            action = 2
        observation, r, done, info = env.step(action)
        obs_list.append(observation[0])
        
        if done == True:
            if observation[0] >= 0.5:
                running = False
        
            elif steps == max_steps:
                running = False
    
    
    reward += max(obs_list)
    reward -= steps
    
    if iteration == 0:
        best_reward = reward
    
    elif reward  > best_reward:
        best_policy = policy
        best_reward = reward
    
     
            
    return best_policy, best_reward
        

def train(iter, environment):
    best_policy = np.random.rand(2)
    best_reward = -1
    
    for i in range(iter):
        print("Episode: ", i + 1)
        best_policy, best_reward = run(i, best_reward, best_policy, environment)
        print("Best policy: " + str(best_policy))
        if i == 0 or i % 200 == 0:
            env = gym.wrappers.Monitor(environment, "/u/lauriek/PythonExercises/OpenAI/MCData", force = True)
            running = True
            observation = env.reset()
            max_steps = 200
            steps = 0
            
            while running:
                steps += 1
                
                if best_policy.dot(observation) <= 0:
                    action = 0
                    
                else:
                    action = 2
                
                observation, reward, done, info = env.step(action)
                
                if done == True:
                    if observation[0] >= 0.5:
                        print("Episode " + str(i + 1) + " number of moves: " +  str(steps))
                        running = False
                
                    
                    elif max_steps == steps:
                        print("Failure")
                        running = False 
        
    return best_policy

best_policy = train(Iterations, env)

def test(policy):
    env = gym.make('MountainCar-v0')
    for i in range(10):
        running = True
        observation = env.reset()
        max_steps = 500
        steps = 0
        while running:
            env.render()
            steps += 1
            if policy.dot(observation) <= 0:
                action = 0
                
            else:
                action = 2
            
            observation, reward, done, info = env.step(action)
            if observation[0] >= 0.5:
                print("Episode " + str(i + 1) + " number of moves: " +  str(steps))
                running = False
            
               
            elif max_steps == steps:
                running = False
test(best_policy)
