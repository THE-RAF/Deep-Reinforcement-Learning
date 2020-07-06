import gym

import matplotlib.pyplot as plt
from matplotlib import style
style.use('seaborn')

import pandas as pd

from monte_carlo_control import *


env = gym.make('CartPole-v0')

observation_upper_bound = [0.25, 1, 0.25, 2]
observation_lower_bound = [-0.2, -1.2, -0.25, -2]
discretization_split = [15, 15, 15, 15]

discretizer = StateDiscretizer(discretization_split=discretization_split)

discretizer.get_boundaries_from_env(environment=env, action_space_size=2, n_episodes=10000)

exploration_strategy = EpsilonGreedyStrategy(100)

agent = MonteCarloAgent(exploration_strategy=exploration_strategy, action_space_size=2, gamma=0.99, state_discretizer=discretizer)

scores = agent.train(env, 10000)

scores_ma = pd.DataFrame(scores).rolling(100).mean()

plt.plot(scores)
plt.plot(scores_ma)

plt.show()
