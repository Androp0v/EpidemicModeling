import numpy as np
import math
import matplotlib.pyplot as plt
import warnings

#Parametros iniciales:

beta = 3.0
mu = 0.1
lamb = 0.4
gamma = 0.0
k = 1.0
T = 20.0
c = 1.0
M = (T+c)

tmax = 200

Sv, Iv, Snv, Inv, qv, qnv, Tvnv, Tnvv, Pv, Pnv, V, I = (np.empty(tmax) for i in range(12))

#Condiciones iniciales:

Sv[0]=1.0
Iv[0]=0.0
Snv[0]=0.0
Inv[0]=0.0

#Funcion de probabilidad:

def f(x):
    if x<0:
        return 0
    else:
        return x/M
    
    #return (1/(1+np.exp(-beta*x)))
    
#Bucle principal de iteracion con ecuaciones SIS:

for i in range(tmax-1):
    V[i]=Sv[i]+Iv[i]
    I[i]=Iv[i]+Inv[i]

    with warnings.catch_warnings():
        
        warnings.filterwarnings('error')

        try:
            Pv[i]=-c-T*Iv[i]/(Iv[i]+Sv[i])
        except Warning:
            Pv[i] = -c -T

        try:
            Pnv[i]=-T*Inv[i]/(Inv[i]+Snv[i])
        except Warning:
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

    
#Gráficos:

plt.plot(V, label = "Vacunados")
plt.plot(I, label = "Infectados")

#Títulos de gráfico y ejes, leyenda:

plt.title("Evolución temporal de la fracción de vacunados e infectados", fontweight = "bold")
plt.xlabel("Número de iteraciones")
plt.ylabel("Fracción respecto al total")
plt.legend()

plt.show()
