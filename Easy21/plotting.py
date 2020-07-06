from matplotlib.pyplot import figure
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

from numpy import arange, meshgrid, zeros


def get_plot_array(q_table):
	plot_value_table = zeros([10, 21])
	for i in range(10):
		for j in range(21):
			plot_value_table[(i, j)] = max([q_table[(i + 1, j + 1, 0)] if (i + 1, j + 1, 0) in q_table else 0,
											q_table[(i + 1, j + 1, 1)] if (i + 1, j + 1, 1) in q_table else 0])

	return plot_value_table

def plot_value_function(q_table):
	value_function = get_plot_array(q_table).transpose()

	# Create x, y coords
	ny, nx = value_function.shape
	cellsize = 1
	x = arange(0, nx, 1) * cellsize
	y = arange(0, ny, 1) * cellsize
	X, Y = meshgrid(x, y)

	# Create matplotlib Figure and Axes
	fig = figure('Easy21 Value Function')
	ax = fig.add_subplot(111, projection='3d')

	# Plot the surface
	ax.plot_surface(X, Y, value_function, cmap=cm.coolwarm, antialiased=True)

	ax.set_xlabel("Dealer's first card")
	ax.set_ylabel("Player's sum")
	ax.set_zlabel("Optimal value function")

def plot_success_rates(success_rates, episodes_x):
	fig = figure(f'Success rate over episodes')
	ax = fig.add_subplot(111)

	ax.plot(episodes_x, success_rates)
	ax.set_xlabel('episodes')
	ax.set_ylabel('success rate')
	
