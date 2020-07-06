import gym

import matplotlib.pyplot as plt
from matplotlib import style
style.use('seaborn')

import torch.nn as nn
from torch.nn.functional import relu

from double_deep_q_learning import DDQNAgent, EpsilonGreedyStrategy, ReplayMemory


class DQN(nn.Module):
	def __init__(self, input_size, output_size):
		super().__init__()

		self.input_size = input_size
		self.output_size = output_size

		self.fc1 = nn.Linear(in_features=input_size, out_features=32)
		self.fc2 = nn.Linear(in_features=32, out_features=32)
		self.out = nn.Linear(in_features=32, out_features=output_size)

	def forward(self, t):
		t = relu(self.fc1(t))
		t = relu(self.fc2(t))
		t = self.out(t)

		return t


strategy = EpsilonGreedyStrategy(start=1, end=0.1, decay=0.001)
replay_memory = ReplayMemory(capacity=1000)

LR = 0.001
GAMMA = 0.99
BATCH_SIZE = 256
SOFT_UPDATE_FACTOR = 0.1

dqn_model = DQN(4, 2)

agent = DDQNAgent(dqn_model=dqn_model,
				   strategy=strategy,
				   replay_memory=replay_memory,
				   learning_rate=LR,
				   gamma=GAMMA,
				   batch_size=BATCH_SIZE,
				   soft_update_factor=SOFT_UPDATE_FACTOR,
				   device='cpu')

env = gym.make('CartPole-v0')

agent.train(env=env, n_episodes=100, render=True)

env.close()

plt.plot(agent.episode_rewards)
plt.show()
