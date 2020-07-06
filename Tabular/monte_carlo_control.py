from collections import namedtuple, deque
from copy import deepcopy
import tqdm

import numpy as np
import random


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


class StateDiscretizer:
	def __init__(self, discretization_split, observation_upper_bound=None, observation_lower_bound=None):
		self.discretization_split = np.array(discretization_split)

		if observation_upper_bound and observation_lower_bound:
			self.observation_upper_bound = np.array(observation_upper_bound)
			self.observation_lower_bound = np.array(observation_lower_bound)
			
			self.discrete_step = (self.observation_upper_bound - self.observation_lower_bound) / self.discretization_split

	def get_boundaries_from_env(self, environment, action_space_size, n_episodes):
		state = environment.reset()

		upper_bound = deepcopy(state)
		lower_bound = deepcopy(state)

		for episode in range(n_episodes):
			state = environment.reset()

			done = False
			while not done:
				action = np.random.randint(0, action_space_size)
				state, _, done, _ = environment.step(action)

				for index, feature in enumerate(state):
					if feature > upper_bound[index]:
						upper_bound[index] = feature

					if feature < lower_bound[index]:
						lower_bound[index] = feature

		self.observation_upper_bound = np.array(upper_bound)
		self.observation_lower_bound = np.array(lower_bound)

		self.discrete_step = (self.observation_upper_bound - self.observation_lower_bound) / self.discretization_split

	def discretize_state(self, state):
		state = np.array(state)

		feature_pct = (state - self.observation_lower_bound) / (self.observation_upper_bound - self.observation_lower_bound)

		discretization_multiplier = np.round(feature_pct * self.discretization_split)
		discrete_state = self.observation_lower_bound + discretization_multiplier * self.discrete_step

		return tuple(discrete_state)


class MonteCarloAgent:
	def __init__(self, exploration_strategy, action_space_size, gamma=1.0, state_discretizer=False):
		self.exploration_strategy = exploration_strategy

		self.GAMMA = gamma
		self.action_space_size = action_space_size

		self.q_table = {}
		self.state_occur_counter = {} 

		self.reset_episode_sample()

		self.state_discretizer = state_discretizer

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

	def train(self, env, episodes, render=False):
		scores = []

		for episode in tqdm.tqdm(range(episodes)):
			state = env.reset()
			state = tuple(state)
			if self.state_discretizer:
				state = self.state_discretizer.discretize_state(state)

			self.reset_episode_sample()

			score = 0

			done = False
			while not done:
				if render:
					env.render()

				action = self.take_action(state)
				next_state, reward, done, info = env.step(action)

				next_state = tuple(next_state)
				if self.state_discretizer:
					next_state = self.state_discretizer.discretize_state(next_state)

				experience = Experience(state, action, reward)
				self.push_experience(experience)

				state = next_state

				score += reward

			self.update_sample_qs()

			scores.append(score)

		print()
		print('training complete')

		return scores
