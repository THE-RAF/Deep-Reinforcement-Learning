import random
from collections import namedtuple, deque
from tqdm import tqdm

import torch
from torch.optim import Adam


Experience = namedtuple('experience', ('state', 'action', 'reward'))


class DPGAgent:
	def __init__(self,
				 state_space_size,
				 action_space_size,
				 network_model,
				 sample_memory_size,
				 sample_batch_size,
				 state_memory_size=1,
				 learning_rate=0.0001,
				 update_policy_every_step=False,
				 device='cpu'):

		self.LEARNING_RATE = learning_rate
		self.SAMPLE_MEMORY_SIZE = sample_memory_size
		self.SAMPLE_BATCH_SIZE = sample_batch_size
		self.STATE_MEMORY_SIZE = state_memory_size

		self.update_policy_every_step = update_policy_every_step

		self.state_memory = deque([[0] * state_space_size] * state_memory_size, maxlen=state_memory_size)

		self.sample_memory = deque(maxlen=sample_memory_size)

		self.device = device

		self.action_space = [i for i in range(action_space_size)]

		self.network_model = network_model(input_size=state_space_size * state_memory_size, output_size=action_space_size)
		self.optimizer = Adam(self.network_model.parameters(), lr=self.LEARNING_RATE)

	def convert_memory_to_state(self):
		state = []
		for memory in self.state_memory:
			state += list(memory)

		return state

	def take_action(self, state):
		with torch.no_grad():
			state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)

			resulting_probabilities = self.network_model(state_tensor)
			resulting_probabilities = resulting_probabilities.tolist()

			return random.choices(self.action_space, resulting_probabilities)[0]

	def update_model_parameters(self, episode_sample, update_policy_every_step=False):
		for i, sample_step in enumerate(episode_sample):
			step_return = 0

			for j in range(i, len(episode_sample)):
				step_return += episode_sample[j].reward

			state_tensor = torch.tensor(sample_step.state, dtype=torch.float32, device=self.device)
			log_policy = torch.log(self.network_model(state_tensor)[sample_step.action])
			to_minimize = -log_policy * step_return

			to_minimize.backward()

			if update_policy_every_step:
				self.optimizer.step()
				self.optimizer.zero_grad()

		if not update_policy_every_step:
			self.optimizer.step()
			self.optimizer.zero_grad()

	def train(self, env, n_episodes, render=False):
		scores = []

		for episode in tqdm(range(n_episodes)):
			score = 0

			env_state = env.reset()
			self.state_memory.append(env_state)

			state = self.convert_memory_to_state()

			done = False

			episode_sample = []

			while not done:
				if render:
					env.render()

				action = self.take_action(state)

				next_env_state, reward, done, info = env.step(action)
				self.state_memory.append(next_env_state)
				next_state = self.convert_memory_to_state()

				experience = Experience(state, action, reward)
				episode_sample.append(experience)

				state = next_state

				score += reward

			self.sample_memory.append(episode_sample)

			if len(self.sample_memory) >= self.SAMPLE_MEMORY_SIZE:
				samples = random.choices(self.sample_memory, k=self.SAMPLE_BATCH_SIZE)

				for sample in samples:
					self.update_model_parameters(sample, self.update_policy_every_step)

			scores.append(score)

		return scores
