import numpy as np
import matplotlib.pyplot as plt
import warnings
import multiprocessing


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
    #if x<0:
    #    return 0
    #else:
    #    return x/M
    
    return (1/(1+np.exp(-beta*x)))

#Image dimensions:

xres = 50
yres = 50
heatmap1 = np.empty((xres,yres))
heatmap2 = np.empty((xres,yres))

#Main loop

indexX = 0
indexY = 0
for gamma in np.linspace(0,1,xres):
	indexX = 0
	for lamb in np.linspace(0,1,yres):
		#Recuperamos las condiciones iniciales:

		Sv = 0.25
		Iv = 0.25
		Snv = 0.25
		Inv = 0.25

		Vmean = 0
		Imean = 0

		for i in range(tmax-1):
		    V = Sv + Iv
		    I = Iv + Inv

		    Vmean += V/tmax
		    Imean += I/tmax

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
		    
		    newSv=(1-Tvnv)*(Sv*(1-qv)+Iv*mu)+Tnvv*(Snv*(1-qv)+Inv*mu)
		    newSnv=Tvnv*(Sv*(1-qnv)+Iv*mu)+(1-Tnvv)*(Snv*(1-qnv)+Inv*mu)
		    newIv=(1-Tvnv)*(Sv*qv+Iv*(1-mu))+Tnvv*(Snv*qv+Inv*(1-mu))
		    newInv=Tvnv*(Sv*qnv+Iv*(1-mu))+(1-Tnvv)*(Snv*qnv+Inv*(1-mu))

		    Sv, Snv, Iv, Inv = newSv, newSnv, newIv, newInv


		Vmean += (Sv+Iv)/tmax
		Imean += (Iv + Inv)/tmax

		heatmap1[indexX, indexY] = Vmean
		heatmap2[indexX, indexY] = Imean

		indexX += 1

	indexY += 1


#Gráficos:

plt.subplot(121)
imageV = plt.imshow(heatmap1, cmap='hot', interpolation='nearest', origin="lower")
plt.colorbar(imageV)
plt.title("Vacunados")
plt.xlabel("Probabilidad de fallo vacuna (γ)")
plt.ylabel("Probabilidad de transmisión (λ)")

plt.subplot(122)
imageI = plt.imshow(heatmap2, cmap='hot', interpolation='nearest', origin="lower")
plt.colorbar(imageI)
plt.title("Infectados")
plt.xlabel("Probabilidad de fallo vacuna (γ)")
plt.ylabel("Probabilidad de transmisión (λ)")

plt.show()
