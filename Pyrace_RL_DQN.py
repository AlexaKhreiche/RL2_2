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

# ==== Config ====
VERSION_NAME = 'DQN_HYBRID'
NUM_EPISODES = 5000
MAX_T = 2000
GAMMA = 0.99
LR = 1e-3
BATCH_SIZE = 64
REPLAY_MEMORY_SIZE = 100_000
BEST_MEMORY_SIZE = 10_000
MIN_REPLAY_SIZE = 1000
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = 0.995

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pygame.init()

# ==== DQN Model ====
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

# ==== Replay Buffer ====
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def store(self, *transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
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

# ==== DQN Agent ====
class DQNAgent:
    def __init__(self, obs_size, n_actions):
        self.model = DQN(obs_size, n_actions).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)
        self.memory = ReplayBuffer(REPLAY_MEMORY_SIZE)
        self.best_memory = ReplayBuffer(BEST_MEMORY_SIZE)

        self.gamma = GAMMA
        self.batch_size = BATCH_SIZE
        self.epsilon = EPSILON_START
        self.best_reward = -float('inf')
        self.episode = 0

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, env.action_space.n - 1)
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            return self.model(state).argmax().item()

    def train(self):
        if len(self.memory) < MIN_REPLAY_SIZE:
            return

        half_batch = self.batch_size // 2
        mem_sample = self.memory.sample(half_batch)
        best_sample = (
            self.best_memory.sample(half_batch)
            if len(self.best_memory) >= half_batch
            else self.memory.sample(half_batch)
        )

        states = torch.cat((mem_sample[0], best_sample[0]))
        actions = torch.cat((mem_sample[1], best_sample[1]))
        rewards = torch.cat((mem_sample[2], best_sample[2]))
        next_states = torch.cat((mem_sample[3], best_sample[3]))
        dones = torch.cat((mem_sample[4], best_sample[4]))

        q_vals = self.model(states).gather(1, actions.unsqueeze(1)).squeeze()
        next_q_vals = self.model(next_states).max(1)[0]
        expected_q = rewards + self.gamma * next_q_vals * (1 - dones)

        loss = F.mse_loss(q_vals, expected_q.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)

    def save(self, path=f"models_{VERSION_NAME}/best_model.pth"):
        torch.save({
            "model_state": self.model.state_dict(),
            "best_memory": list(self.best_memory.buffer),
            "episode": self.episode,
            "epsilon": self.epsilon,
            "best_reward": self.best_reward,
        }, path)

    def load(self, path=f"models_{VERSION_NAME}/best_model.pth"):
        if os.path.exists(path):
            data = torch.load(path, map_location=device, weights_only=False)
            self.model.load_state_dict(data["model_state"])
            self.best_memory.buffer = deque(data.get("best_memory", []), maxlen=BEST_MEMORY_SIZE)
            self.episode = data.get("episode", 0)
            self.epsilon = data.get("epsilon", EPSILON_START)
            self.best_reward = data.get("best_reward", -float('inf'))
            self.model.eval()
            print(f"✅ Model loaded from {path}")

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
            action = agent.act(state)
            next_state, reward, done, _, info = env.step(action)

            # Reward shaping
            if "dist" in info: reward += 3 * info["dist"]
            if "crash" in info: reward -= 500
            if "check" in info: reward += 500

            agent.memory.store(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            t += 1

            # 👀 Proper rendering & messages
            env.set_msgs([
                '🚗 SIMULATING',
                f'Episode: {ep}',
                f'Time steps: {t}',
                f'Reward: {total_reward:.0f}'
            ])
            env.render()
            pygame.event.pump()
            time.sleep(0.01)

            if t % 2 == 0:
                agent.train()

        print(f"[Episode {ep}] Reward: {total_reward:.1f}")
        rewards_log.append(total_reward)

        # Stuck detection
        if total_reward == agent.best_reward:
            stuck_count += 1
        else:
            stuck_count = 0

        # Save high-reward transitions
        if total_reward > agent.best_reward * 0.9:
            for i in list(agent.memory.buffer)[-t:]:
                agent.best_memory.store(*i)

        # Save best model
        if total_reward > agent.best_reward:
            print("🎯 New best model!")
            agent.best_reward = total_reward
            agent.save()

        if stuck_count >= 3:
            print("🌀 Agent stuck! Resetting epsilon.")
            agent.epsilon = EPSILON_START

        agent.episode += 1

        if ep % 100 == 0:
            plt.plot(rewards_log)
            plt.ylabel("Reward")
            plt.xlabel("Episode")
            plt.title("Training Progress")
            plt.savefig(f"models_{VERSION_NAME}/rewards.png")
            plt.close()

# ==== Entrypoint ====
if __name__ == "__main__":
    if not os.path.exists(f"models_{VERSION_NAME}"):
        os.makedirs(f"models_{VERSION_NAME}")

    env = gym.make("Pyrace-v1").unwrapped
    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n

    agent = DQNAgent(obs_size, n_actions)
    agent.load()

    try:
        simulate(env, agent, NUM_EPISODES)
    except KeyboardInterrupt:
        print("🛑 Interrupted. Saving model...")
        agent.save()
