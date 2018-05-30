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
beta = 0.03
epsilon = 10**(-18)

Nsteps = 100

#Image dimensions:

xres = 10
yres = 10

#Helper functions:

def prob(x):
    if x < 0:
        return 0
    else:
        return x/M

    #return 0.5*(1+beta*x)

    #return (1/(1 + np.exp(-beta*x)))

#Funcion simulacion:

def simulation(arguments):

	gamma = arguments[0]
	lamb = arguments[1]

	#Initialize Sv, Iv, Snv, Inv and network:

	network = nx.read_edgelist("Red300Nodos.net")

	Sv = 0
	Iv = 0
	Snv = 0
	Inv = 0

	#Provide initial values for all nodes in the network:

	for i in range(1,len(network)+1):
		network.node[str(i)]['Health'] = random.choice(['Healthy', 'Infected'])
		network.node[str(i)]['Vaccination'] = random.choice(['Vaccinated', 'Not-vaccinated'])

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

	#Normalization of Sv, Iv, Snv, Inv:

	Sv /= 300
	Iv /= 300
	Snv /= 300
	Inv /= 300

	#Measurement loop:	

	Vmean = 0
	Imean = 0		

	for k in range(0, Nsteps):

		#Payoffs and strategy:

		Pv = -c -T*(Iv/(Iv + Sv + epsilon))
		Pnv = -T*(Inv/(Inv + Snv + epsilon))

		Tvnv = prob(Pnv - Pv)
		Tnvv = prob(Pv - Pnv)

		for i in range(1,len(network)+1):

			#Change vaccination strategies:

			if network.node[str(i)]['Vaccination'] == 'Vaccinated' and random.random() < Tvnv:
				network.node[str(i)]['Vaccination'] = 'Not-vaccinated'

			elif network.node[str(i)]['Vaccination'] == 'Not-vaccinated' and random.random() < Tnvv:
				network.node[str(i)]['Vaccination'] = 'Vaccinated'

			#Infection dynamics:

			if network.node[str(i)]['Health'] == 'Infected' and random.random() < mu: #If it's infected, probability it becomes healthy
				network.node[str(i)]['Health'] = 'Healthy'

			else: #If it's healthy, probability it becomes infected
				for neighbor in network.neighbors(str(i)):

					if network.node[neighbor]['Health'] == 'Infected' and network.node[str(i)]['Health'] == 'Healthy':

						if network.node[str(i)]['Vaccination'] == 'Not-vaccinated' and random.random() < lamb:
							network.node[str(i)]['Health'] = 'Infected'
							break
						elif network.node[str(i)]['Vaccination'] == 'Vaccinated' and random.random() < gamma and random.random() < lamb:
							network.node[str(i)]['Health'] = 'Infected'
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

		#Normalization of Sv, Iv, Snv, Inv:

		Sv /= 300
		Iv /= 300
		Snv /= 300
		Inv /= 300

		Vmean += Sv + Iv
		Imean += Iv + Inv

	Vmean /= Nsteps
	Imean /= Nsteps

	return(Vmean,Imean)

#Main loop:

if __name__ == '__main__':

	pool = multiprocessing.Pool(processes = multiprocessing.cpu_count())

	input = []

	for gamma in np.linspace(0,1,yres):
		for lamb in np.linspace(0,1,xres):
			input.append((gamma,lamb))

	results = pool.map(simulation, input)

	pool.close()

	resultsV = np.empty(xres*yres)
	resultsI = np.empty(xres*yres)

	l = 0
	for i in results:
		resultsV[l] = i[0]
		resultsI[l] = i[1]
		l += 1

	heatmap1 = np.reshape(resultsV, (yres, xres))
	heatmap2 = np.reshape(resultsI, (yres, xres))

	#Save numpy array to text file:

	np.savetxt("/Users/Raul/Desktop/Vacunados.txt", heatmap1)
	np.savetxt("/Users/Raul/Desktop/Infectados.txt", heatmap2)


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