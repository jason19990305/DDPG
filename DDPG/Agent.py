
from torch.distributions import Normal
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np 
import torch
import time

# Custom class
from DDPG.ReplayBuffer import ReplayBuffer
from DDPG.ActorCritic import Actor , Critic



class Agent():
    def __init__(self,args,env,hidden_layer_num_list=[64,64]):

        # Hyperparameter
        self.evaluate_freq_steps = args.evaluate_freq_steps
        self.max_train_steps = args.max_train_steps
        self.num_actions = args.num_actions
        self.batch_size = args.batch_size
        self.num_states = args.num_states
        self.mem_min = args.mem_min
        self.gamma = args.gamma
        self.set_var = args.var
        self.var = self.set_var
        self.tau = args.tau
        self.lr = args.lr

        # Variable
        self.total_steps = 0
        self.training_count = 0
        self.evaluate_count = 0

        # other
        self.env = env
        self.action_max = env.action_space.high[0]
        self.replay_buffer = ReplayBuffer(args)

        # Actor-Critic
        self.actor = Actor(args,hidden_layer_num_list.copy())
        self.critic = Critic(args,hidden_layer_num_list.copy())
        self.actor_target =  Actor(args,hidden_layer_num_list.copy())
        self.critic_target =  Critic(args,hidden_layer_num_list.copy())
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr, eps=1e-5)
        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr, eps=1e-5)

        print(self.actor)
        print(self.critic)
        print("-----------")

    def choose_action(self,state):

        state = torch.tensor(state, dtype=torch.float)

        s = torch.unsqueeze(state,0)
        with torch.no_grad():
            a = self.actor(s)
            a = Normal(a,self.var).sample()            
            a = torch.clamp(a,-1,1)
            
        return a.cpu().numpy().flatten() 

    def evaluate_action(self,state):

        state = torch.tensor(state, dtype=torch.float)
        s = torch.unsqueeze(state,0)

        with torch.no_grad():
            a = self.actor(s)     
        return a.cpu().numpy().flatten()

    def evaluate_policy(self, env , render = False):
        times = 10
        evaluate_reward = 0
        for i in range(times):
            s, info = env.reset()
            
            done = False
            episode_reward = 0
            while True:
                a = self.evaluate_action(s)  # We use the deterministic policy during the evaluating
            
                s_, r, done, truncted, _ = env.step(a)

               
                episode_reward += r
                s = s_
                #print(episode_reward)
                if truncted or done:
                    break
            evaluate_reward += episode_reward

        return evaluate_reward / times

    def var_decay(self, total_steps):
        new_var = self.set_var * (1 - total_steps / self.max_train_steps)
        self.var = new_var + 10e-10
        
    def train(self):
        time_start = time.time()
        episode_reward_list = []
        episode_count_list = []
        episode_count = 0
        
        # Training Loop
        while self.total_steps < self.max_train_steps:
            s = self.env.reset()[0]            
            while True:
                a = self.choose_action(s)
                s_, r, done , truncated , _ = self.env.step(a)
                done = done or truncated


                # storage data
                self.replay_buffer.store(s, a, [r], s_, done)
                
                # update state
                s = s_

                if self.replay_buffer.count >= self.mem_min:
                    self.training_count += 1
                    self.update()

                if self.total_steps % self.evaluate_freq_steps == 0:
                    self.evaluate_count += 1
                    evaluate_reward = self.evaluate_policy(self.env)
                    episode_reward_list.append(evaluate_reward)
                    episode_count_list.append(episode_count)
                    time_end = time.time()
                    h = int((time_end - time_start) // 3600)
                    m = int(((time_end - time_start) % 3600) // 60)
                    second = int((time_end - time_start) % 60)
                    print("---------")
                    print("Time : %02d:%02d:%02d"%(h,m,second))
                    print("Training episode : %d\tStep : %d / %d"%(episode_count,self.total_steps,self.max_train_steps))
                    print("Evaluate count : %d\tEvaluate reward : %0.2f"%(self.evaluate_count,evaluate_reward))

                self.total_steps += 1
                if done or truncated:
                    break
            episode_count += 1
        # Plot the training curve
        plt.plot(episode_count_list, episode_reward_list)
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Training Curve")
        plt.show()
            
    def update(self):
        s, a, r, s_, done = self.replay_buffer.numpy_to_tensor()  # Get training data .type is tensor    

        index = np.random.choice(len(r),self.batch_size,replace=False)

        # Get minibatch
        minibatch_s = s[index]
        minibatch_a = a[index]
        minibatch_r = r[index]
        minibatch_s_ = s_[index]
        minibatch_done = done[index]

        # update Actor
        action = self.actor(minibatch_s)
        value = self.critic(minibatch_s,action)
        actor_loss = -torch.mean(value)
        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5) # Trick : Clip grad
        self.optimizer_actor.step()


        # Update Critic
        next_action = self.actor_target(minibatch_s_)
        next_value = self.critic_target(minibatch_s_,next_action)
        v_target = minibatch_r + self.gamma * next_value * (1 - minibatch_done)

        value = self.critic(minibatch_s,minibatch_a)
        critic_loss = F.mse_loss(value,v_target)
        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5) # Trick : Clip grad
        self.optimizer_critic.step()

        
        self.var_decay(total_steps=self.total_steps)
        
        # Update target networks
        self.soft_update(self.critic_target,self.critic, self.tau)
        self.soft_update(self.actor_target, self.actor, self.tau)    

    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
            
    
        