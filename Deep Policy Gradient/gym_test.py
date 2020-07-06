import gym

import matplotlib.pyplot as plt
from matplotlib import style
style.use('seaborn')

from state_memory_DPG import DPGAgent

import torch.nn as nn
from torch.nn.functional import relu, softmax


class Network(nn.Module):
	def __init__(self, input_size, output_size):
		super().__init__()

		self.input_size = input_size
		self.output_size = output_size

		self.fc1 = nn.Linear(in_features=input_size, out_features=64)
		self.fc2 = nn.Linear(in_features=64, out_features=64)
		self.out = nn.Linear(in_features=64, out_features=output_size)

	def forward(self, t):
		t = relu(self.fc1(t))
		t = relu(self.fc2(t))
		t = softmax(self.out(t), dim=0)

		return t


model = Network

env = gym.make('CartPole-v0')

agent = DPGAgent(state_space_size=4,
				 action_space_size=2,
				 network_model=model,
				 sample_memory_size=20,
				 sample_batch_size=10,
				 state_memory_size=2,
				 learning_rate=0.001,
				 update_policy_every_step=True,
				 device='cpu')

scores = agent.train(env, n_episodes=300, render=False)

plt.plot(scores)
plt.show()

agent.train(env, n_episodes=100, render=True)

env.close()
