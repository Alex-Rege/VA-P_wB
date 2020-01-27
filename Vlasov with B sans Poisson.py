#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 01:36:59 2019

@author: sidd
"""

    #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 10:45:21 2019

@author: sidd
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 10:28:55 2019

@author: sidd
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import scipy
from numpy import linalg
from random import randrange
import copy


import numba
from numba import jit
from numba import njit
import time

from scipy.interpolate import CubicSpline
from scipy import interpolate


#fréquence cyclotronique
omega_c=0.1

#mode de Fourier
n=50
#n=randrange(1,100)

#choix de m pour les vecteurs propres
m=1
#m=randrange(1,100)


flag=2

if omega_c!=0:
    flag=-1
    


#position
N=16
k=0.4
L=2*np.pi
dx=L/(N-1)
X=np.linspace(0,L,N,endpoint=True)
Xbis=np.linspace(0,L,N-1,endpoint=False)
#print(X,Xbis)

#vitesse 1
M1=16
A1=10
dv1=2*A1/(M1-1)
V1=np.linspace(-A1,A1,M1,endpoint=True)


#vitesse 2
M2=16
A2=10
dv2=2*A2/(M2-1)
V2=np.linspace(-A2,A2,M2,endpoint=True)

#temps
T=10
mod=1
modbis=100
dt=0.1
Nbt=int(T/dt)
Time=np.array([i*dt for i in range(Nbt)])



print ('--------------')
print ('dx =',dx)
print ('dv1=',dv1)
print ('dv2=',dv2)
print ('Flag (donnée initiale)= ',flag)
print ('--------------')


def f_mode_propre(omega_c):
    def u(v1,v2):
        tau_p=1/(1+v1*v1+v2*v2)
        u_n=np.exp(-n*1j*v2/omega_c)
        return tau_p*u_n
    
    f2V=np.vectorize(u)
    ff2=f2V(np.transpose([V1]),V2)
    
    
    ffx=[np.exp(n*1j*x)*ff2 for x in X]
    return ffx

def f_mode_propre1(omega_c):
    def u(v1,v2):
        return (np.exp(-v1*v1/2)*np.exp(-v2*v2/2))
    
    f2V=np.vectorize(u)
    ff2=f2V(np.transpose([V1]),V2)
    
    
    ffx=[ff2 for x in X]
    return ffx

def fnon_mode_propre(omega_c):
    def vec_propre(v1,v2):
        tau_p=np.exp(-np.sqrt(v1*v1+v2*v2))
        u_n=np.exp(-2*np.pi*n*1j*v2/omega_c)
        if v2<=0 and v1==0:
          e_m=-np.pi/2
        elif v2>=0 and v1==0:
            e_m=np.pi/2
        elif v2>=0 and v1>=0:
            e_m=np.exp(m*1j*np.arctan(v2/v1))
        elif v2<=0 and v1>=0:
            e_m=2*np.pi+np.exp(m*1j*np.arctan(v2/v1))
        else:
            e_m=np.pi+np.exp(m*1j*np.arctan(v2/v1))
        return tau_p*u_n*e_m
    
    f2V=np.vectorize(vec_propre)
    ff2=f2V(np.transpose([V1]),V2)
    
    
    ffx=[np.exp(n*1j*x)*ff2 for x in X]
    return ffx

f=fnon_mode_propre(omega_c)


def SLC(f,X,V1,V2):
    f1=copy.deepcopy(f)
    X1=copy.deepcopy(X)
    V1bis=copy.deepcopy(V1)
    V2bis=copy.deepcopy(V2)
    #calcul masse initiale
    m_ini=0
    for k in range(N-1):
      for i in range(M1):
        for j in range(M2):
            m_ini+=f1[k][i,j]
    real_m_ini=m_ini*dx*dv1*dv2
    l=[0]
    for t in range(1,Nbt):
       start=time.time()
       if (t % mod==0):
         print("Temps=",t*dt,"et Niter=",t)
         #affichage masse initiale
         print("   masse initiale= ",real_m_ini/L)
    
         #affichage masse à chaque instant
         masse_non_norma=0
         for k in range(N-1):
          for i in range(M1-1):
           for j in range(M2-1):
              masse_non_norma+=f1[k][i,j]
         masse=masse_non_norma*dx*dv1*dv2
         print("   masse courante= ",masse/L)
         print("   Erreur relative de masse= ", (masse-real_m_ini)/real_m_ini)
         
         mf=[max(np.abs(np.max(f1[i])),np.abs(np.min(f1[i]))) for i in range(len(f1))]
         mff=max(mf)
         l.append(mff)
         print("   Max Densité u= ",mff)
         print("   n=",n )
         """
         p=randrange(M1)
         print("   p=",p)
         
         plt.figure(t)
         plt.figure(figsize= (15, 10))
         plt.plot(V2,[f1[0][p,j] for j in range(M2)],'x', label="f au temps "+str(t*dt)+" selon la direction "+str(p))
         
         plt.xlabel('v2')
         plt.ylabel('f selon v2')
         plt.legend()
         
         """
       #résolution à v1 et v2 constants
       for i in range(M1):
          Xp=X1-V1bis[i]*dt
          
          #périodisation des pieds de caractéristiques
          Xp=Xp-X1[0]
          Xp=[math.fmod(x,X1[-1]-X1[0]) for x in Xp]
          for p in range(len(Xp)):
              if Xp[p]<0:
                  Xp[p]+=X1[-1]
              else:
                  Xp[p]+=X1[0]
            
          """
          """
          
          #Xp=Xp+(Xp<0)*X1[-1]-(Xp>X1[-1])*X1[-1]
          #print(Xp)
          for j in range (M2):  
            fij=[f1[k][i,j] for k in range(N)]
            fij[-1]=fij[0]
          
            #interpolation par splines cubiques
            a=scipy.interpolate.CubicSpline(X1,fij,bc_type='periodic')(Xp)
            for k in range(N):
               f1[k][i,j]=a[k]
       """    
       masse_non_norma=0
       for k in range(N-1):
          for i in range(M1-1):
           for j in range(M2-1):
              masse_non_norma+=f1[k][i,j]
       masse=masse_non_norma*dx*dv1*dv2
       print("     masse courante phase 1= ",masse/L)
       print("     Erreur relative de masse phase 1= ", (masse-real_m_ini)/real_m_ini)
       """
       #résolution à x et v2 constants  
       for k in range(N):
           for j in range(M2):
               V1bisp=V1bis+omega_c*V2bis[j]*dt
               
               #périodisation des pieds de caractéristiques
               V1bisp=V1bisp-V1bis[0]
               V1bisp=[math.fmod(x,V1bis[-1]-V1bis[0]) for x in V1bisp]
               for p in range(len(V1bisp)):
                   if V1bisp[p]<0:
                       V1bisp[p]+=V1bis[-1]
                   else:
                       V1bisp[p]+=V1bis[0]
               """
               """
               #V1bisp=V1bisp+(V1bisp<V1bis[0])*(V1bis[-1]-V1bis[0])-(V1bisp>V1bis[-1])*(V1bis[-1]-V1bis[0])

               fkj=[f1[k][i,j] for i in range(M1)]
               fkj[0]=fkj[-1]
            
               """
               for i in range(M1):
                  if (V1bisp[i]>=V1[-1] or V1bisp[i]<=V1[0]):
                    fkj[i]=0       
               
               """
               
               #interpolation par spline cubique
               #a=scipy.interpolate.CubicSpline(V1bis,fkj,bc_type='clamped')(V1bisp)
               a=scipy.interpolate.CubicSpline(V1bis,fkj,bc_type='periodic')(V1bisp)               
               f1[k][:,j:j+1]=np.transpose(np.array([[a[i] for i in range(M1)]]))
       """
       masse_non_norma=0
       for k in range(N-1):
          for i in range(M1-1):
           for j in range(M2-1):
              masse_non_norma+=f1[k][i,j]
       masse=masse_non_norma*dx*dv1*dv2
       print("     masse courante phase 2= ",masse/L)
       print("     Erreur relative de masse phase 2= ", (masse-real_m_ini)/real_m_ini)
       """
       #résolution à x et v1 constants    
       for k in range(N):
           
           
           for i in range(M1):
               V2bisp=V2bis-omega_c*V1bis[i]*dt
               #périodisation des pieds de caractéristiques
               V2bisp=V2bisp-V2bis[0]
               V2bisp=[math.fmod(x,V2bis[-1]-V2bis[0]) for x in V2bisp]
               for p in range(len(V2bisp)):
                   if V2bisp[p]<0:
                       V2bisp[p]+=V2bis[-1]
                   else:
                       V2bisp[p]+=V2bis[0]
               
               #V2bisp=V2bisp+(V2bisp<V2bis[0])*(V2bis[-1]-V2bis[0])-(V2bisp>V2bis[-1])*(V2bis[-1]-V2bis[0])
               fki=[f1[k][i,j] for j in range(M2)]
               fki[0]=fki[-1]
               
               """
               for j in range(M2):
                  if (V2bisp[j]>=V2[-1] or V2bisp[j]<=V2[0]):
                    fki[j]=0     
               """
               
               #interpolation par spline cubique
               a=scipy.interpolate.CubicSpline(V2bis,fki,bc_type='periodic')(V2bisp)
               f1[k][i:i+1,:]=np.array([[a[j] for j in range(M2)]])
       """
       masse_non_norma=0
       for k in range(N-1):
          for i in range(M1-1):
           for j in range(M2-1):
              masse_non_norma+=f1[k][i,j]
       masse=masse_non_norma*dx*dv1*dv2
       print("     masse courante phase 3= ",masse/L)
       print("     Erreur relative de masse phase 3= ", (masse-real_m_ini)/real_m_ini)
       """
       end=time.time()
       print("   Elapsed for one iteration = %s" % (end - start))
    
    return 

start = time.time()    

fend=SLC(f,X,V1,V2)

end = time.time()
print("Elapsed (after compilation) = %s" % (end - start))

    
    
    


