import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import multiprocessing
import random
from matplotlib.ticker import FormatStrFormatter
import sys

#Initial parameters:

mu = 0.1

T = 20.0
c = 1.0

M = (T+c)
beta = 3.0
epsilon = 10**(-18)

Nterm = 60
Nsteps = 50

#Image dimensions:

xres = 10
yres = 10

#Helper functions:

def prob(x):
    #if x < 0:
    #    return 0
    #else:
    #    return x/M

    #return 0.5*(1+beta*x)

    return (1/(1 + np.exp(-beta*x)))

#Funcion simulacion:

def simulation(arguments):

	gamma = arguments[1]
	lamb = arguments[0]

	#Initialize Sv, Iv, Snv, Inv and network:

	#network = nx.read_edgelist("Red300Nodos.net")

	NumberOfNodes = 9600
	network = nx.cycle_graph(NumberOfNodes)

	Sv = 0
	Iv = 0
	Snv = 0
	Inv = 0

	#Provide initial values for all nodes in the network:

	for node in network:

		network.node[node]['Health'] = random.choice(['Healthy', 'Infected'])

		network.node[node]['Vaccination'] = random.choice(['Vaccinated', 'Not-vaccinated'])


	#Compute initial Sv, Iv, Snv, Inv values: 

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

	Sv /= NumberOfNodes
	Snv /= NumberOfNodes
	Iv /= NumberOfNodes
	Inv /= NumberOfNodes

	#Thermalization loop:

	for i in range(Nterm):

		#Calculate payoffs:

		try:
			Pv = -c -T*Iv/(Iv + Sv)
		except ZeroDivisionError:
			Pv = -c -T

		try:
			Pnv = -T*Inv/(Inv + Snv)
		except ZeroDivisionError:
			Pnv = -T

		Tvnv = prob(Pnv - Pv)
		Tnvv = prob(Pv - Pnv)
	
		#Change vaccination strategies:

		for node in network:

			if network.node[node]['Vaccination'] == 'Vaccinated': 
				if random.random() < Tvnv:
					network.node[node]['Vaccination'] = 'Not-vaccinated'

			elif network.node[node]['Vaccination'] == 'Not-vaccinated':
				if random.random() < Tnvv:
					network.node[node]['Vaccination'] = 'Vaccinated'

		#Infection dynamics:

		for node in network:

			if network.node[node]['Health'] == 'Infected': 
				if random.random() < mu: #If it's infected, probability it becomes healthy
					network.node[node]['Health'] = 'Healthy'

			else: #If it's healthy, probability it becomes infected

				if network.node[node]['Vaccination'] == 'Vaccinated':
					if random.random() < gamma:
						for neighbor in network.neighbors(node):
							if network.node[neighbor]['Health'] == 'Infected' and random.random() < lamb:
				 				network.node[node]['Health'] = 'Infected'
				 				break
				else:
					for neighbor in network.neighbors(node):
						if network.node[neighbor]['Health'] == 'Infected' and random.random() < lamb:
				 			network.node[node]['Health'] = 'Infected'
				 			break


				# for neighbor in network.neighbors(node):

				# 	if network.node[neighbor]['Health'] == 'Infected':

				# 		if network.node[node]['Vaccination'] == 'Not-vaccinated':
				# 			if random.random() < lamb:
				# 				network.node[node]['Health'] = 'Infected'
				# 				break
				# 		elif network.node[node]['Vaccination'] == 'Vaccinated':
				# 			if random.random() < gamma and random.random() < lamb:
				# 				network.node[node]['Health'] = 'Infected'
				# 				break

		#Update Sv, Iv, Snv, Iv:

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

		Sv /= NumberOfNodes
		Snv /= NumberOfNodes
		Iv /= NumberOfNodes
		Inv /= NumberOfNodes

	#Measurement loop:

	Vmean = 0
	Imean = 0

	for i in range(Nsteps):

		#Calculate payoffs:

		try:
			Pv = -c -T*Iv/(Iv + Sv)
		except ZeroDivisionError:
			Pv = -c -T

		try:
			Pnv = -T*Inv/(Inv + Snv)
		except ZeroDivisionError:
			Pnv = -T

		Tvnv = prob(Pnv - Pv)
		Tnvv = prob(Pv - Pnv)
	
		#Change vaccination strategies:

		for node in network:

			if network.node[node]['Vaccination'] == 'Vaccinated': 
				if random.random() < Tvnv:
					network.node[node]['Vaccination'] = 'Not-vaccinated'

			elif network.node[node]['Vaccination'] == 'Not-vaccinated':
				if random.random() < Tnvv:
					network.node[node]['Vaccination'] = 'Vaccinated'

		#Infection dynamics:

		for node in network:

			if network.node[node]['Health'] == 'Infected': 
				if random.random() < mu: #If it's infected, probability it becomes healthy
					network.node[node]['Health'] = 'Healthy'

			else: #If it's healthy, probability it becomes infected

				if network.node[node]['Vaccination'] == 'Vaccinated':
					if random.random() < gamma:
						for neighbor in network.neighbors(node):
							if network.node[neighbor]['Health'] == 'Infected' and random.random() < lamb:
				 				network.node[node]['Health'] = 'Infected'
				 				break
				else:
					for neighbor in network.neighbors(node):
						if network.node[neighbor]['Health'] == 'Infected' and random.random() < lamb:
				 			network.node[node]['Health'] = 'Infected'
				 			break


				# for neighbor in network.neighbors(node):

				# 	if network.node[neighbor]['Health'] == 'Infected':

				# 		if network.node[node]['Vaccination'] == 'Not-vaccinated':
				# 			if random.random() < lamb:
				# 				network.node[node]['Health'] = 'Infected'
				# 				break
				# 		elif network.node[node]['Vaccination'] == 'Vaccinated':
				# 			if random.random() < gamma and random.random() < lamb:
				# 				network.node[node]['Health'] = 'Infected'
				# 				break

		#Update Sv, Iv, Snv, Iv:

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

		Sv /= NumberOfNodes
		Snv /= NumberOfNodes
		Iv /= NumberOfNodes
		Inv /= NumberOfNodes

		Vmean += Sv + Iv
		Imean += Iv + Inv

	#Calculate and return means:

	Vmean /= Nsteps
	Imean /= Nsteps

	return(Vmean, Imean)






#Main loop:

if __name__ == '__main__':

	pool = multiprocessing.Pool(processes = multiprocessing.cpu_count())

	input = []

	for gamma in np.linspace(0,1,yres):
		for lamb in np.linspace(0,1,xres):
			input.append((gamma,lamb))

	results = pool.map(simulation, input)
	#results2 = pool.map(simulation, input)
	#results3 = pool.map(simulation, input)
	# results4 = pool.map(simulation, input)
	# results5 = pool.map(simulation, input)

	pool.close()

	resultsV = np.empty(xres*yres)
	resultsI = np.empty(xres*yres)
	resultsV2 = np.empty(xres*yres)
	resultsI2 = np.empty(xres*yres)
	resultsV3 = np.empty(xres*yres)
	resultsI3 = np.empty(xres*yres)

	l = 0
	for i in results:
		resultsV[l] = i[0]
		resultsI[l] = i[1]
		l += 1

	# l = 0
	# for i in results2:
	# 	resultsV2[l] = i[0]
	# 	resultsI2[l] = i[1]
	# 	l += 1

	# l = 0
	# for i in results3:
	# 	resultsV3[l] = i[0]
	# 	resultsI3[l] = i[1]
	# 	l += 1

	# l = 0
	# for i in results4:
	# 	resultsV[l] += i[0]
	# 	resultsI[l] += i[1]
	# 	l += 1

	# l = 0
	# for i in results5:
	# 	resultsV[l] += i[0]
	# 	resultsI[l] += i[1]
	# 	l += 1

	# resultsV /= 5
	# resultsI /= 5

	heatmap1 = np.reshape(resultsV, (yres, xres))
	heatmap2 = np.reshape(resultsI, (yres, xres))

	# heatmap12 = np.reshape(resultsV2, (yres, xres))
	# heatmap22 = np.reshape(resultsI2, (yres, xres))

	# heatmap13 = np.reshape(resultsV3, (yres, xres))
	# heatmap23 = np.reshape(resultsI3, (yres, xres))

	#Save numpy array to text file:

	np.savetxt("/Users/Raul/Desktop/Vacunados.txt", heatmap1)
	np.savetxt("/Users/Raul/Desktop/Infectados.txt", heatmap2)

	# np.savetxt("/Users/Raul/Desktop/Vacunados2.txt", heatmap12)
	# np.savetxt("/Users/Raul/Desktop/Infectados2.txt", heatmap22)

	# np.savetxt("/Users/Raul/Desktop/Vacunados3.txt", heatmap13)
	# np.savetxt("/Users/Raul/Desktop/Infectados3.txt", heatmap23)


	#Gráficos:

	nxticks = 6
	nyticks = 6
	ncolorticks = 10

	ax1 = plt.subplot(121)
	imageV = plt.imshow(heatmap1, cmap='hot', interpolation='nearest', origin="lower")
	plt.clim(0,1)
	plt.colorbar(imageV)
	plt.title("Vacunados")
	plt.xlabel("Probabilidad de fallo vacuna (γ)")
	plt.ylabel("Probabilidad de transmisión (λ)")

	ax1.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
	xvalues = np.linspace(-0.5,xres-0.5,nxticks)
	xlabels = [(x+0.5)/xres for x in xvalues]
	plt.xticks(xvalues, xlabels)

	ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
	yvalues = np.linspace(-0.5,yres-0.5,nyticks)
	ylabels = [(y+0.5)/yres for y in yvalues]
	plt.yticks(yvalues, ylabels)

	ax2 = plt.subplot(122)
	imageI = plt.imshow(heatmap2, cmap='hot', interpolation='nearest', origin="lower")
	plt.clim(0,1)
	plt.colorbar(imageI)
	plt.title("Infectados")
	plt.xlabel("Probabilidad de fallo vacuna (γ)")
	plt.ylabel("Probabilidad de transmisión (λ)")

	ax2.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
	plt.xticks(xvalues, xlabels)

	ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
	plt.yticks(yvalues, ylabels)

	plt.show()
