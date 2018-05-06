#cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True

import warnings
import numpy as np
from libc.math cimport exp

#Parámetros iniciales:
cdef double beta = 3.0
cdef double mu = 0.1
cdef double k = 1.0
cdef double T = 20.0
cdef double c = 1.0
cdef double M = (T + c) 
cdef double Sv = 0.25
cdef double Iv = 0.25
cdef double Snv = 0.25
cdef double Inv = 0.25
cdef double newSv = 0
cdef double newIv = 0
cdef double newSnv = 0
cdef double newInv = 0

cdef int tmax = 800
cdef int tTermal = 200

cdef double f(double x):
    if x < 0:
        return 0
    else:
        return x/M

    #return (1/(1 + exp(-beta*x)))

cpdef cSimulation(double lamb, double gamma):
    
    Sv = 0.25
    Iv = 0.25
    Snv = 0.25
    Inv = 0.25

    cdef double Vmean = 0
    cdef double Imean = 0

    cdef int i

    for i in range(0,tTermal):
         V = Sv + Iv
         I = Iv + Inv

         with warnings.catch_warnings():
                
            warnings.filterwarnings('error')

            try:
                Pv = -c -T*Iv/(Iv+Sv)
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

            newSv, newSnv, newIv, newInv = (1-Tvnv)*(Sv*(1-qv)+Iv*mu)+Tnvv*(Snv*(1-qv)+Inv*mu), Tvnv*(Sv*(1-qnv)+Iv*mu)+(1-Tnvv)*(Snv*(1-qnv)+Inv*mu), (1-Tvnv)*(Sv*qv+Iv*(1-mu))+Tnvv*(Snv*qv+Inv*(1-mu)), Tvnv*(Sv*qnv+Iv*(1-mu))+(1-Tnvv)*(Snv*qnv+Inv*(1-mu))
            
            Sv, Snv, Iv, Inv = newSv, newSnv, newIv, newInv

    for i in range(0,tmax - tTermal - 1):
         V = Sv + Iv
         I = Iv + Inv

         Vmean += V
         Imean += I

         with warnings.catch_warnings():
                
            warnings.filterwarnings('error')

            try:
                Pv = -c -T*Iv/(Iv+Sv)
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

            newSv, newSnv, newIv, newInv = (1-Tvnv)*(Sv*(1-qv)+Iv*mu)+Tnvv*(Snv*(1-qv)+Inv*mu), Tvnv*(Sv*(1-qnv)+Iv*mu)+(1-Tnvv)*(Snv*(1-qnv)+Inv*mu), (1-Tvnv)*(Sv*qv+Iv*(1-mu))+Tnvv*(Snv*qv+Inv*(1-mu)), Tvnv*(Sv*qnv+Iv*(1-mu))+(1-Tnvv)*(Snv*qnv+Inv*(1-mu))
            
            Sv, Snv, Iv, Inv = newSv, newSnv, newIv, newInv

    Vmean += (Sv+Iv)
    Imean += (Iv + Inv)

    Vmean = Vmean / (tmax - tTermal)
    Imean = Imean / (tmax - tTermal)

    return(Vmean, Imean)
