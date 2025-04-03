
#importing all the necessary libraries
import os, math, random, time, json
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import gym_race
import pygame

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque

# step 1: configuring the environment
VERSION_NAME = 'Models_DQN' #folder where the model will be saved
NUM_EPISODES = 10000 #number of episodes to train the model
MAX_T = 2000 #max time steps per episode
GAMMA = 0.99 #discount factor
LR = 1e-3 #learning rate
BATCH_SIZE = 64 #batch size for training
REPLAY_MEMORY_SIZE = 100_000 #size of the replay memory
BEST_MEMORY_SIZE = 10_000 #size of the best memory
MIN_REPLAY_SIZE = 1000 #minimum size of the replay memory to start training
EPSILON_START = 1.0 #initial epsilon for epsilon-greedy policy
EPSILON_END = 0.1 #final epsilon for epsilon-greedy policy
EPSILON_DECAY = 0.995 #decay rate for epsilon

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #using GPU if available
pygame.init()

# Step 2: defining the DQN neural network
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# step 3: defining the replay buffer to store the experiences
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity) #ring buffer to store the experiences

    def store(self, *transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        #randomly sampling a batch of experiences from the buffer
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(states).to(device),
            torch.LongTensor(actions).to(device),
            torch.FloatTensor(rewards).to(device),
            torch.FloatTensor(next_states).to(device),
            torch.FloatTensor(dones).to(device),
        )

    def __len__(self):
        return len(self.buffer)

#steo 4: defining the DQN agent
class DQNAgent:
    def __init__(self, obs_size, n_actions):
        #initializing the NN and optimizer
        self.model = DQN(obs_size, n_actions).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)
        
        #initializing the replay buffer
        self.memory = ReplayBuffer(REPLAY_MEMORY_SIZE)
        self.best_memory = ReplayBuffer(BEST_MEMORY_SIZE)

        #initializing the hyperparameters & statr tracking
        self.gamma = GAMMA
        self.batch_size = BATCH_SIZE
        self.epsilon = EPSILON_START
        self.best_reward = -float('inf')
        self.episode = 0

    def act(self, state):
        #slecting an action using epsilon-greedy policy
        if random.random() < self.epsilon:
            return random.randint(0, env.action_space.n - 1) #exploration
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(device) #exploitation
            return self.model(state).argmax().item()

    def train(self):
        #we dont train the model until we have enough samples in the replay buffer
        if len(self.memory) < MIN_REPLAY_SIZE:
            return

        # splitting the batch btw normal and high-reward experiences
        half_batch = self.batch_size // 2
        mem_sample = self.memory.sample(half_batch)
        best_sample = (
            self.best_memory.sample(half_batch)
            if len(self.best_memory) >= half_batch
            else self.memory.sample(half_batch)
        )

        #combining both samples for training
        states = torch.cat((mem_sample[0], best_sample[0]))
        actions = torch.cat((mem_sample[1], best_sample[1]))
        rewards = torch.cat((mem_sample[2], best_sample[2]))
        next_states = torch.cat((mem_sample[3], best_sample[3]))
        dones = torch.cat((mem_sample[4], best_sample[4]))

        #definding the update rule for the DQN
        q_vals = self.model(states).gather(1, actions.unsqueeze(1)).squeeze()
        next_q_vals = self.model(next_states).max(1)[0]
        expected_q = rewards + self.gamma * next_q_vals * (1 - dones)

        loss = F.mse_loss(q_vals, expected_q.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        #using a decay schedule for epsilon
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)

    def save(self, path=f"models_{VERSION_NAME}/best_model.pth"):
        #saving the model and the memory state
        torch.save({
            "model_state": self.model.state_dict(),
            "best_memory": list(self.best_memory.buffer),
            "episode": self.episode,
            "epsilon": self.epsilon,
            "best_reward": self.best_reward,
        }, path)

    def load(self, path=f"models_{VERSION_NAME}/best_model.pth"):
        #loading the saved model and the memory state
        if os.path.exists(path):
            data = torch.load(path, map_location=device, weights_only=False)
            self.model.load_state_dict(data["model_state"])
            self.best_memory.buffer = deque(data.get("best_memory", []), maxlen=BEST_MEMORY_SIZE)
            self.episode = data.get("episode", 0)
            self.epsilon = data.get("epsilon", EPSILON_START)
            self.best_reward = data.get("best_reward", -float('inf'))
            self.model.eval()
            print(f"✅ Model loaded from {path}")

#step 5: the main simulation / training loop

def simulate(env, agent, episodes):
    stuck_count = 0
    rewards_log = []

    for ep in range(agent.episode, episodes):
        state, _ = env.reset()
        env.set_view(True)
        env.pyrace.mode = 2

        env.set_msgs(['🏁 DQN Agent Starting...'])
        env.render()
        pygame.event.pump()
        time.sleep(0.5)

        total_reward, t = 0, 0
        done = False

        while not done and t < MAX_T:
            action = agent.act(state) #selecting an action using the policy
            next_state, reward, done, _, info = env.step(action) #taking a step in the environment

            # reward shaping based on the environment feedback
            if "dist" in info: reward += 3 * info["dist"]
            if "crash" in info: reward -= 500
            if "check" in info: reward += 500
            
            #storing the experience in the replay buffer
            agent.memory.store(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            t += 1

            # updating the environment display
            env.set_msgs([
                'SIMULATING',
                f'Episode: {ep}',
                f'Time steps: {t}',
                f'Reward: {total_reward:.0f}'
            ])
            env.render()
            pygame.event.pump()
            time.sleep(0.01)

            #training the model every 2 steps
            if t % 2 == 0:
                agent.train()

        print(f"[Episode {ep}] Reward: {total_reward:.1f}")
        rewards_log.append(total_reward)

        # detecting if the agent is stuck
        if total_reward == agent.best_reward:
            stuck_count += 1
        else:
            stuck_count = 0

        # saving the high-reward transitions/episodes into the best memory
        if total_reward > agent.best_reward * 0.9:
            for i in list(agent.memory.buffer)[-t:]:
                agent.best_memory.store(*i)

        # saving the best model
        if total_reward > agent.best_reward:
            print("🎯 New best model!")
            agent.best_reward = total_reward
            agent.save()

        # resetting the epsilon if the agent is stuck for too long
        if stuck_count >= 3:
            print(" the agent stuck :( Resetting epsilon.")
            agent.epsilon = EPSILON_START

        agent.episode += 1


# step 6: the main function:
if __name__ == "__main__":
    if not os.path.exists(f"models_{VERSION_NAME}"):
        os.makedirs(f"models_{VERSION_NAME}")

    #loadig the Pyrace-v1 gym environment
    env = gym.make("Pyrace-v1").unwrapped
    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n

    #initializing the DQN agent
    agent = DQNAgent(obs_size, n_actions)
    agent.load()

    #starting the training loop
    try:
        simulate(env, agent, NUM_EPISODES)
    except KeyboardInterrupt:
        print("🛑 Interrupted. Saving model...")
        agent.save()
