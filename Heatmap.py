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

xres = 100
yres = 100
heatmap1 = np.empty((xres,yres))
heatmap2 = np.empty((xres,yres))

#Main loop

indexX = 0
indexY = 0
for gamma in np.linspace(0,1,xres):

	indexY = 0

	for lamb in np.linspace(0,1,yres):

		#Recuperamos las condiciones iniciales:

		Sv[0]=0.25
		Iv[0]=0.25
		Snv[0]=0.25
		Inv[0]=0.25

		for i in range(tmax-1):
		    V[i]=Sv[i]+Iv[i]
		    I[i]=Iv[i]+Inv[i]

		    with warnings.catch_warnings():
		        
		        warnings.filterwarnings('error')

		        try:
		            Pv[i]=-c-T*Iv[i]/(Iv[i]+Sv[i])
		        except Warning:
		        	print("Warning")
		        	Pv[i] = -c -T

		        try:
		            Pnv[i]=-T*Inv[i]/(Inv[i]+Snv[i])
		        except Warning:
		        	print("Warning")
		        	Pnv[i] = -T


		    Tvnv[i]=f(Pnv[i]-Pv[i])
		    Tnvv[i]=f(Pv[i]-Pnv[i])

		    qv[i]=1-(1-lamb*gamma*(gamma*Iv[i]+Inv[i]))**k
		    qnv[i]=1-(1-lamb*(gamma*Iv[i]+Inv[i]))**k
		    
		    Sv[i+1]=(1-Tvnv[i])*(Sv[i]*(1-qv[i])+Iv[i]*mu)+Tnvv[i]*(Snv[i]*(1-qv[i])+Inv[i]*mu)
		    Iv[i+1]=Tvnv[i]*(Sv[i]*(1-qnv[i])+Iv[i]*mu)+(1-Tnvv[i])*(Snv[i]*(1-qnv[i])+Inv[i]*mu)
		    Snv[i+1]=(1-Tvnv[i])*(Sv[i]*qv[i]+Iv[i]*(1-mu))+Tnvv[i]*(Snv[i]*qv[i]+Inv[i]*(1-mu))
		    Inv[i+1]=Tvnv[i]*(Sv[i]*qnv[i]+Iv[i]*(1-mu))+(1-Tnvv[i])*(Snv[i]*qnv[i]+Inv[i]*(1-mu))

		V[tmax-1]=Sv[tmax-1]+Iv[tmax-1]
		I[tmax-1]=Iv[tmax-1]+Inv[tmax-1]

		Vmean = np.mean(V[tTermal:])
		Imean = np.mean(I[tTermal:])

		heatmap1[indexX, indexY] = Vmean
		heatmap2[indexX, indexY] = Imean

		indexY += 1

	indexX += 1


plt.subplot(121)
imageV = plt.imshow(heatmap1, cmap='hot', interpolation='nearest')
plt.colorbar(imageV)
plt.title("Vacunados")
plt.xlabel("gamma")
plt.ylabel("lambda")

plt.subplot(122)
imageI = plt.imshow(heatmap2, cmap='hot', interpolation='nearest')
plt.colorbar(imageI)
plt.title("Infectados")
plt.xlabel("gamma")
plt.ylabel("lambda")

plt.show()