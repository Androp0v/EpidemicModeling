import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
from matplotlib.ticker import FormatStrFormatter
import UpdateCoefficients
from UpdateCoefficients import *
import sys

#Image dimensions:

xres = 1000
yres = 1000

#Funcion simulacion:

def simulation(arguments):

	return(cSimulation(arguments[0], arguments[1]))

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
