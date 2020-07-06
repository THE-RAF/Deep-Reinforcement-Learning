import random
from math import exp
from collections import namedtuple, deque
from copy import deepcopy

import numpy as np

import torch
from torch.optim import Adam
from torch.nn.functional import mse_loss


class EpsilonGreedyStrategy:
	def __init__(self, start, end, decay):
		self.start = start
		self.end = end
		self.decay = decay

		self.current_step = 0

	def get_exploration_rate(self):
		exploration_rate = self.end + (self.start - self.end) * exp(-1 * self.current_step * self.decay)
		self.current_step += 1

		return exploration_rate


Experience = namedtuple('experience', ('state', 'action', 'reward', 'new_state', 'done'))


class ReplayMemory:
	def __init__(self, capacity):
		self.memory = deque(maxlen=capacity)

	def push(self, experience):
		self.memory.append(experience)

	def sample(self, batch_size):
		return random.sample(self.memory, batch_size)

	def unpacked_sample(self, batch_size):
		sample = self.sample(batch_size)
		return Experience(*zip(*sample))

	def can_provide_sample(self, batch_size):
		return len(self.memory) >= batch_size


class QValues:
	@staticmethod
	def get_action_qs(network, states, actions):
		q_values = network(states).gather(dim=1, index=actions.unsqueeze(dim=1))
		return q_values.flatten()

	@staticmethod
	def get_qs_argmax(network, next_states):
		with torch.no_grad():
			values = network(next_states).argmax(dim=1)
			return values


def soft_update(target_net, policy_net, tau):
	for param, other_net_param in zip(target_net.parameters(), policy_net.parameters()):
		param.data.copy_(tau * other_net_param.data + (1.0 - tau) * param.data)


class DDQNAgent:
	def __init__(self, dqn_model, strategy, replay_memory, learning_rate, gamma, batch_size, soft_update_factor, device):
		self.device = device

		self.GAMMA = gamma
		self.BATCH_SIZE = batch_size
		self.SOFT_UPDATE_FACTOR = soft_update_factor

		self.strategy = strategy
		self.replay_memory = replay_memory

		self.policy_network = deepcopy(dqn_model)

		self.target_network = deepcopy(dqn_model)
		self.target_network.load_state_dict(self.policy_network.state_dict())
		self.target_network.eval()

		self.policy_network.to(self.device)
		self.target_network.to(self.device)

		self.optimizer = Adam(self.policy_network.parameters(), lr=learning_rate)

		self.episode_rewards = []

	def take_action(self, state, exploration_rate=0):
		if np.random.uniform() < exploration_rate:
			action = np.random.randint(0, self.policy_network.output_size)

		else:
			with torch.no_grad():
				prediction = self.policy_network(torch.tensor(state, dtype=torch.float32, device=self.device))
				action = torch.argmax(prediction).item()

		return action

	def update_networks(self):
		states, actions, rewards, new_states, dones = self.replay_memory.unpacked_sample(self.BATCH_SIZE)

		states_tensor = torch.tensor(states, dtype=torch.float32, device=self.device)
		new_states_tensor = torch.tensor(new_states, dtype=torch.float32, device=self.device)
		actions_tensor = torch.tensor(actions, device=self.device)
		rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)
		dones_tensor = torch.tensor(dones, dtype=torch.float32, device=self.device)

		current_q_values = QValues.get_action_qs(self.policy_network, states_tensor, actions_tensor)
		
		next_actions_tensor = QValues.get_qs_argmax(self.policy_network, new_states_tensor)
		next_q_values = QValues.get_action_qs(self.target_network, new_states_tensor, next_actions_tensor)

		target_q_values = (next_q_values * self.GAMMA) * dones_tensor + rewards_tensor

		loss = mse_loss(current_q_values, target_q_values)

		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()

		soft_update(self.target_network, self.policy_network, tau=self.SOFT_UPDATE_FACTOR)


	def train(self, env, n_episodes, render=False):
		for episode in range(n_episodes):
			print('EPISODE', episode)
			state = env.reset()
			done = False

			episode_reward = 0
			while not done:
				if render:
					env.render()

				exploration_rate = self.strategy.get_exploration_rate()
				action = self.take_action(state, exploration_rate=exploration_rate)
				new_state, reward, done, _ = env.step(action)
				done_int = 0 if done else 1

				experience = Experience(state, action, reward, new_state, done_int)
				self.replay_memory.push(experience)

				if self.replay_memory.can_provide_sample(self.BATCH_SIZE):
					self.update_networks()
					
				state = new_state
				episode_reward += reward

			self.episode_rewards.append(episode_reward)
