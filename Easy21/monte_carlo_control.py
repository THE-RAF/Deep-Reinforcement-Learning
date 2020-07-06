from environment import Easy21Env

from collections import namedtuple, deque
import tqdm

import numpy as np
import random
from math import exp

import matplotlib.pyplot as plt
from matplotlib import style
import plotting
style.use('seaborn')


Experience = namedtuple('experience', ('state', 'action', 'reward'))


class EpsilonGreedyStrategy:
	def __init__(self, n0=100):
		self.N0 = n0
		self.state_counter = {}

	def get_exploration_rate(self, state):
		if state in self.state_counter:
			self.state_counter[state] += 1

		else:
			self.state_counter[state] = 0

		return self.N0 / (self.N0 + self.state_counter[state])


class ExpEpsilonGreedyStrategy:
	def __init__(self, start_epsilon, end_epsilon, decay):
		self.start_epsilon = start_epsilon
		self.end_epsilon = end_epsilon
		self.decay = decay

		self.step_counter = 0

	def get_exploration_rate(self):
		self.step_counter += 1

		return self.end_epsilon + (self.start_epsilon - self.end_epsilon) * exp(-self.step_counter * self.decay)


class MonteCarloAgent:
	def __init__(self, exploration_strategy, action_space_size, gamma=1.0):
		self.exploration_strategy = exploration_strategy

		self.GAMMA = gamma
		self.action_space_size = action_space_size

		self.q_table = {}
		self.state_occur_counter = {} 

		self.reset_episode_sample()

	def reset_episode_sample(self):
		self.episode_sample = []

	def take_action(self, state):
		exploration_rate = self.exploration_strategy.get_exploration_rate(state)
		if np.random.uniform() < exploration_rate:
			return np.random.randint(0, self.action_space_size)

		else:
			q_value_indexes = [tuple(state) + (action,) for action in range(self.action_space_size)]
			q_values = [self.q_table[index] if index in self.q_table else 0 for index in q_value_indexes]

			return np.argmax(q_values)
		
	def push_experience(self, experience):
		self.episode_sample.append(experience)

	def update_sample_qs(self):
		for i in range(len(self.episode_sample)):

			sample_return = 0
			for j in range(i, len(self.episode_sample)):
				gamma_power = j - i
				sample_return += (self.GAMMA**gamma_power) * self.episode_sample[j].reward

			q_value_index = self.episode_sample[i].state + (self.episode_sample[i].action,)

			if q_value_index not in self.q_table:
				self.state_occur_counter[q_value_index] = 0
				self.q_table[q_value_index] = 0

			self.state_occur_counter[q_value_index] += 1
			self.q_table[q_value_index] += (1/self.state_occur_counter[q_value_index]) * (sample_return - self.q_table[q_value_index])


def convert_experience(experience):
	if experience.action == 'hit':
		action = 0
	if experience.action == 'stick':
		action = 1

	return Experience((experience.state[0], experience.state[1]), action, experience.reward)

def convert_agent_action(action):
	if action == 0:
		converted_action = 'hit'
	if action == 1:
		converted_action = 'stick'

	return converted_action


env = Easy21Env(print_game=False)

exploration_strategy = EpsilonGreedyStrategy(n0=50)
agent = MonteCarloAgent(exploration_strategy=exploration_strategy, action_space_size=env.action_space_size, gamma=0.5)

last_episodes_rewards = deque(maxlen=1000)
success_rates = []
episodes_x = []

for episode in tqdm.tqdm(range(200000)):
	state = env.reset()

	agent.reset_episode_sample()

	done = False
	while not done:
		action = convert_agent_action(agent.take_action(state))

		reward, next_state = env.step(action)

		experience = Experience(state, action, reward)
		agent.push_experience(convert_experience(experience))

		if next_state == 'terminal':
			done = True

		else:
			state = next_state

	agent.update_sample_qs()

	last_episodes_rewards.append(reward)

	if episode % last_episodes_rewards.maxlen == 0:
		success_rates.append(last_episodes_rewards.count(1) / last_episodes_rewards.maxlen)
		episodes_x.append(episode)

success_rates.pop(0)
episodes_x.pop(0)

plotting.plot_success_rates(success_rates=success_rates, episodes_x=episodes_x)

plotting.plot_value_function(q_table=agent.q_table)

plt.show()
