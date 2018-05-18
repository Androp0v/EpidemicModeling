import networkx as nx 
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random

network = nx.read_edgelist("Red300Nodos.net")
positions = nx.spring_layout(network)

#Initial parameters:

mu = 0.1
lamb = 0.1
gamma = 0.25


#Helper functions:

def prob(x):
    if x<0:
        return 0
    else:
        return x/M

#Initial conditions:
node_colors = []
infectedCount = [0]
healthyCount = [0]

for i in range(1,len(network)+1):
	network.node[str(i)]['Health'] = random.choice(['Healthy', 'Healthy'])

	if network.node[str(i)]['Health'] == 'Healthy':
		node_colors.append('g')
		healthyCount[0] += 1
	else:
		node_colors.append('r')
		infectedCount[0] += 1

	network.node[str(i)]['Vaccination'] = random.choice(['Vaccinated', 'Not-vaccinated'])

network.node['1']['Health'] = 'Infected'
network.node['2']['Health'] = 'Infected'
network.node['3']['Health'] = 'Infected'

#Updating network

def update_network(frame):

	update_plot(frame)

	for i in range(1,len(network)):

		if network.node[str(i)]['Health'] == 'Infected' and random.random() < mu: #If it's infected, probability it becomes healthy
			network.node[str(i)]['Health'] = 'Healthy'
			node_colors[i - 1] = 'g'

		else: #If it's healthy, probability it becomes infected
			for neighbor in network.neighbors(str(i)):

				if network.node[neighbor]['Health'] == 'Infected' and network.node[str(i)]['Health'] == 'Healthy':

					if network.node[str(i)]['Vaccination'] == 'Not-vaccinated' and random.random() < lamb:
						network.node[str(i)]['Health'] = 'Infected'
						node_colors[i - 1] = 'r'
						break
					elif network.node[str(i)]['Vaccination'] == 'Vaccinated' and random.random() < gamma and random.random() < lamb:
						network.node[str(i)]['Health'] = 'Infected'
						node_colors[i - 1] = 'r'
						break
					if network.node[str(i)]['Health'] == 'Infected':
						print("Already infected node!")

	infectedCountTemp = 0
	healthyCountTemp = 0
	for node in network:
		if network.node[node]['Health'] == 'Healthy':
			healthyCountTemp += 1
		else:
			infectedCountTemp += 1
	infectedCount.append(infectedCountTemp)
	healthyCount.append(healthyCountTemp)


#Plotting

figure, (networkPlot, linePlot) = plt.subplots(1,2)

def update_plot(frame):
	plt.cla()
	nx.draw(network, pos = positions, ax = networkPlot, node_color = node_colors, edgecolors = 'black', node_size = 85)
	plt.plot(infectedCount, color = 'r')
	plt.plot(healthyCount, color = 'g')
	networkPlot.set_position([0, 0, 0.5, 1]) #Fill axis area

animation = FuncAnimation(figure, update_network, interval = 1)

plt.show()