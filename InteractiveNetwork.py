import networkx as nx 
import matplotlib.pyplot as plt
import numpy as np
import random
from matplotlib.animation import FuncAnimation
from math import log, exp, log10

network = nx.read_edgelist("Red300Nodos.net")
positions = nx.spring_layout(network)

#Initial parameters:

mu = 0.1
lamb = 0.5
gamma = 0.5

T = 20.0
c = 1.0

M = (T+c)
beta = 0.03
epsilon = 10**(-18)


#Helper functions:

def prob(x):
    if x<0:
        return 0
    else:
        return x/M

    #return (1/(1 + exp(-beta*x)))

#Initial conditions:
node_colors = []
node_borders = []
node_sizes = []

infectedCount = []
healthyCount = []
vaccinatedCount = []
TvnvCount = []
TnvvCount = []

Sv = 0
Iv = 0
Snv = 0
Inv = 0


#Initial graph:

for i in range(1,len(network)+1):
	network.node[str(i)]['Health'] = random.choice(['Healthy', 'Healthy', 'Infected'])

	if network.node[str(i)]['Health'] == 'Healthy':
		node_colors.append('g')
	else:
		node_colors.append('r')

	network.node[str(i)]['Vaccination'] = random.choice(['Vaccinated', 'Not-vaccinated'])

	if network.node[str(i)]['Vaccination'] == 'Vaccinated':
		node_borders.append('b')
	else:
		node_borders.append('black')

	node_sizes.append(log(network.degree[str(i)]) * 50)


for node in network:
	if network.node[node]['Health'] == 'Healthy':
		if network.node[node]['Vaccination'] == 'Vaccinated':
			Sv += 1
		else:
			Snv += 1
	else:
		if network.node[node]['Vaccination'] == 'Not-vaccinated':
			Inv += 1
		else:
			Iv += 1

#Updating network

def update_network(frame):

	update_plot(frame)

	#Payoffs and strategy:

	global Sv, Iv, Snv, Inv

	Pv = -c -T*(Iv/(Iv + Sv + epsilon))
	Pnv = -T*(Inv/(Inv + Snv + epsilon))

	Tvnv = prob(Pnv - Pv)
	Tnvv = prob(Pv - Pnv)

	for i in range(1,len(network)+1):

		#Change vaccination strategies:

		if network.node[str(i)]['Vaccination'] == 'Vaccinated' and random.random() < Tvnv:
			network.node[str(i)]['Vaccination'] = 'Not-vaccinated'
			node_borders[i-1] = 'black'

		elif network.node[str(i)]['Vaccination'] == 'Not-vaccinated' and random.random() < Tnvv:
			network.node[str(i)]['Vaccination'] = 'Vaccinated'
			node_borders[i-1] = 'b'

		#Infection dynamics:

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


	#Update Sv, Iv, Snv, Inv to calculate payoffs in next iteration:

	Sv, Iv, Snv, Inv = (0,0,0,0)

	for node in network:
		if network.node[node]['Health'] == 'Healthy':
			if network.node[node]['Vaccination'] == 'Vaccinated':
				Sv += 1
			else:
				Snv += 1
		else:
			if network.node[node]['Vaccination'] == 'Not-vaccinated':
				Inv += 1
			else:
				Iv += 1

	#Add data for line plots:

	infectedCount.append(Iv + Inv)
	healthyCount.append(Sv + Snv)
	vaccinatedCount.append(Iv + Sv)
	TvnvCount.append(Tvnv)
	TnvvCount.append(Tnvv)


#Plotting

figure, (row1, row2) = plt.subplots(2,2)
networkPlot = row1[0]
linePlot = row1[1]
linePlot2 = row2[1]
row2[0].axis('off') #Hide plot axis over network

#figure, networkPlot = plt.subplots(1,1)

def update_plot(frame):
	#plt.cla()
	nx.draw(network, pos = positions, ax = networkPlot, node_color = node_colors, edgecolors = node_borders, node_size = node_sizes, width = 0.5)
	
	linePlot.plot(infectedCount, color = 'r')
	linePlot.plot(healthyCount, color = 'g')
	linePlot.plot(vaccinatedCount, color = 'b')
	linePlot.set_ylim((0,300))

	linePlot2.plot(TvnvCount, color = 'black', linewidth = 1)
	linePlot2.plot(TnvvCount, color = 'b', linewidth = 1)
	#linePlot2.set_ylim((0,1))

	networkPlot.set_position([0, 0, 0.5, 1]) #Fill axis area (with subplot)
	#networkPlot.set_position([0, 0, 1, 1]) #Fill axis area (alone)

animation = FuncAnimation(figure, update_network, interval = 1)

plt.show()
