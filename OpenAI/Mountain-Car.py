import random, numpy as np, gym
from gym import wrappers
env = gym.make('MountainCar-v0')


Iterations = 10000

best_policy = None 



def run(iteration, best_reward, best_policy, environment):
    max_steps = 200
    steps = 0
    policy = np.random.rand(2)  
    observation = env.reset()
    running = True
    obs_list = []
    while running:
        #env.render()
        steps += 1
        if policy.dot(observation) <= 0:
            action = 0
                  
        else:
            action = 2
        observation, reward, done, info = env.step(action)
         
        if done == True:
            obs_list.append(observation[0])
            reward += max(obs_list)
               
            if steps < best_reward or iteration == 0:
                best_policy = policy
                best_reward = steps
               
                running = False
        if steps == max_steps:
            
            running = False
    
     
            
    return best_policy, best_reward
        

def train(iter, environment):
    best_policy = None 
    best_reward = -1
    
    for i in range(iter):
        print("Episode: ", i + 1)
        best_policy, best_reward = run(i, best_reward, best_policy, environment)
        print(best_policy)
        
        if i < 10 or i % 500 == 0:
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
    
