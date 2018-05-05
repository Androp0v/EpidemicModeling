import numpy as np
import matplotlib.pyplot as plt
import warnings
import multiprocessing
from matplotlib.ticker import FormatStrFormatter

#Parametros iniciales:

beta = 3.0
mu = 0.1
k = 1.0
T = 20.0
c = 1.0
M = (T+c)

tmax = 100
tTermal = 50

Sv, Iv, Snv, Inv, qv, qnv, Tvnv, Tnvv, Pv, Pnv, V, I = (np.empty(tmax) for i in range(12))

#Funcion de probabilidad:

def f(x):
    if x<0:
        return 0
    else:
        return x/M
    
    #return (1/(1+np.exp(-beta*x)))

#Image dimensions:

xres = 100
yres = 100

#Funcion simulacion:

def simulation(arguments):

	lamb = arguments[0]
	gamma = arguments[1]

	Sv = 0.25
	Iv = 0.25
	Snv = 0.25
	Inv = 0.25

	Vmean = 0
	Imean = 0

	for i in range(tmax-1):
	    V = Sv + Iv
	    I = Iv + Inv

	    Vmean += V
	    Imean += I

	    with warnings.catch_warnings():
	        
	        warnings.filterwarnings('error')

	        try:
	            Pv = -c-T*Iv/(Iv+Sv)
	        except Warning:
	        	Pv = -c -T

	        try:
	            Pnv = -T*Inv/(Inv+Snv)
	        except Warning:
	        	Pnv = -T


	    Tvnv = f(Pnv - Pv)
	    Tnvv = f(Pv - Pnv)

	    qv=1-(1-lamb*gamma*(gamma*Iv+Inv))**k
	    qnv=1-(1-lamb*(gamma*Iv+Inv))**k
	    
	    newSv = (1-Tvnv)*(Sv*(1-qv)+Iv*mu)+Tnvv*(Snv*(1-qv)+Inv*mu)
	    newSnv = Tvnv*(Sv*(1-qnv)+Iv*mu)+(1-Tnvv)*(Snv*(1-qnv)+Inv*mu)
	    newIv = (1-Tvnv)*(Sv*qv+Iv*(1-mu))+Tnvv*(Snv*qv+Inv*(1-mu))
	    Inv = Tvnv*(Sv*qnv+Iv*(1-mu))+(1-Tnvv)*(Snv*qnv+Inv*(1-mu))

	    Sv, Snv, Iv = newSv, newSnv, newIv


	Vmean += (Sv+Iv)
	Imean += (Iv + Inv)

	Vmean /= tmax
	Imean /= tmax

	return((Vmean, Imean))



#Main loop
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
