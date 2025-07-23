
import gymnasium as gym # openai gym
import numpy as np 
import argparse
from DDPG.Agent import Agent



class main():
    def __init__(self,args):
        env_name = 'Pendulum-v1'
        env = gym.make('Pendulum-v1')
        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.shape[0]
        print(num_actions)
        print(num_states)
        
        # args
        args.num_actions = num_actions
        args.num_states = num_states


        # print args 
        print("---------------")
        for arg in vars(args):
            print(arg,"=",getattr(args, arg))
        print("---------------")

        # create agent
        hidden_layer_num_list = [256,256]
        agent = Agent(args , env , hidden_layer_num_list)

        # trainning
        agent.train() 

        # evaluate 
        render_env = gym.make(env_name,render_mode='human')
        
        for i in range(10000):
            evaluate_reward = agent.evaluate_policy(render_env)
            print(f"Evaluate Episode {i+1}: Average Reward = {evaluate_reward:.2f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameters Setting for DDPG")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate of actor")
    parser.add_argument("--var", type=float, default=3, help="Normal noise var")
    parser.add_argument("--tau", type=float, default=0.001, help="Parameter for soft update")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--mem_min", type=float, default=100, help="minimum size of replay memory before updating actor-critic.")

    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--buffer_size", type=int, default=int(10000), help="Learning rate of actor")
    parser.add_argument("--max_train_steps", type=int, default=int(3e4), help=" Maximum number of training steps")
    parser.add_argument("--evaluate_freq_steps", type=float, default=2e3, help="Evaluate the policy every 'evaluate_freq' steps")
    args = parser.parse_args()

    main(args)