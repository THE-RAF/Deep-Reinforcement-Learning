from environment import Easy21Env

from collections import namedtuple, deque
import tqdm

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import style
import plotting
style.use('seaborn')


Transition = namedtuple('transition', ('state', 'action', 'reward', 'next_state', 'next_action', 'done'))


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


class SarsaLambdaAgent:
	def __init__(self, exploration_strategy, action_space_size, learning_rate, lambda_, gamma=1.0):
		self.exploration_strategy = exploration_strategy

		self.LEARNING_RATE = learning_rate
		self.LAMBDA = lambda_
		self.GAMMA = gamma
		self.action_space_size = action_space_size

		self.q_table = {}
		self.eligibility_traces = {}

	def reset_eligibility_traces(self):
		for eligibility_trace in self.eligibility_traces:
			self.eligibility_traces[eligibility_trace] = 0

	def take_action(self, state, act_greedly=False):
		exploration_rate = self.exploration_strategy.get_exploration_rate(state)

		if np.random.uniform() < exploration_rate:
			return np.random.randint(0, self.action_space_size)

		else:
			q_value_indexes = [tuple(state) + (action,) for action in range(self.action_space_size)]
			q_values = [self.q_table[index] if index in self.q_table else 0 for index in q_value_indexes]

			return np.argmax(q_values)

	def step(self, transition):
		current_q_index = transition.state + (transition.action,)

		if current_q_index not in self.q_table:
			self.q_table[current_q_index] = 0
			self.eligibility_traces[current_q_index] = 0

		current_q = self.q_table[current_q_index]

		if not transition.done:
			next_q_index = transition.next_state + (transition.next_action,)

			if next_q_index not in self.q_table:
				self.q_table[next_q_index] = 0
				self.eligibility_traces[next_q_index] = 0

			next_q = self.q_table[next_q_index]

			td_error = transition.reward + self.GAMMA * next_q - current_q

		else:
			td_error = transition.reward - current_q

		self.eligibility_traces[current_q_index] += 1

		for q_index in self.q_table:
			self.q_table[q_index] += self.LEARNING_RATE * td_error * self.eligibility_traces[q_index]
			self.eligibility_traces[q_index] *= self.GAMMA * self.LAMBDA


def convert_agent_action(action):
	if action == 0:
		converted_action = 'hit'
	if action == 1:
		converted_action = 'stick'

	return converted_action

env = Easy21Env(print_game=False)

exploration_strategy = EpsilonGreedyStrategy(n0=50)

agent = SarsaLambdaAgent(exploration_strategy=exploration_strategy, action_space_size=2, learning_rate=0.01, lambda_=0.3, gamma=0.3)

last_episodes_rewards = deque(maxlen=1000)
success_rates = []
episodes_x = []

for episode in tqdm.tqdm(range(20000)):
	agent.reset_eligibility_traces()

	state = env.reset()
	action = agent.take_action(state)

	done = False
	while not done:
		reward, next_state = env.step(convert_agent_action(action))
		next_action = agent.take_action(state)

		if next_state == 'terminal':
			done = True

		transition = Transition(state, action, reward, next_state, next_action, done)
		agent.step(transition)

		if not done:
			state = next_state
			action = next_action

	last_episodes_rewards.append(reward)

	if episode % last_episodes_rewards.maxlen == 0:
		success_rates.append(last_episodes_rewards.count(1) / last_episodes_rewards.maxlen)
		episodes_x.append(episode)

success_rates.pop(0)
episodes_x.pop(0)

plotting.plot_value_function(agent.q_table)

plt.show()
