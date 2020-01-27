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
import os

import numba
from numba import jit
from numba import njit
import time

from scipy.interpolate import CubicSpline
from scipy import interpolate


"""
flag=2 Landau Damping
flag=1 Landau Damping avec B
flag=0 Double faisceaux
flag=-1 Double faisceaux avec B
"""
flag=1

#perturbation
epsilon= 0.001

#perturbation pour les modes propres
epsilon1=0.001

#fréquence cyclotronique
omega_c=0.1
if flag==2 or flag==0:
    omega_c=0

#mode de Fourier
n=randrange(1,100)

#choix de m pour les vecteurs propres
m=randrange(1,100)

    
#vitesse double faisceaux
v0=2.4

#position
N=32
k=0.4
L=2*np.pi/k
dx=L/(N-1)
X=np.linspace(0,L,N,endpoint=True)
Xbis=np.linspace(0,L,N-1,endpoint=False)
#print(X,Xbis)

#vitesse 1
M1=96
A1=10
dv1=2*A1/(M1-1)
V1=np.linspace(-A1,A1,M1,endpoint=True)


#vitesse 2
M2=96
A2=10
dv2=2*A2/(M2-1)
V2=np.linspace(-A2,A2,M2,endpoint=True)

#temps
T=200
mod=1
modbis=100
dt=50
Nbt=int(T/dt)
Time=np.array([i*dt for i in range(Nbt)])



print ('--------------')
print ('dx =',dx)
print ('dv1=',dv1)
print ('dv2=',dv2)
print ('dt=',dt)
print ('Flag (donnée initiale)= ',flag)
print ('--------------')


def f_ini(epsilon,k):
    if (flag==2 or flag==1):
     def f1(v1,v2):
            return 1/(2*np.pi)*(np.exp(-v1*v1/2)*np.exp(-v2*v2/2))
     f1V=np.vectorize(f1)
     ff=f1V(np.transpose([V1]),V2)
    elif(flag==0 or flag==-1):
     def f3(v1,v2):
         a=np.exp((-(v1-v0)**2)/2)*np.exp((-(v2-v0)**2)/2)
         b=np.exp((-(v1+v0)**2)/2)*np.exp((-(v2+v0)**2)/2)
         return 1/(2*2*np.pi)*(a+b)
     f3V=np.vectorize(f3)
     ff=f3V(np.transpose([V1]),V2)
    ffx=[(1+epsilon*np.cos(k*x))*ff for x in X]
    return ffx

def f_mode_propre(epsilon,k,omega_c):
    def f0(v1,v2):
        return 1/(2*np.pi)*(np.exp(-v1*v1/2)*np.exp(-v2*v2/2))
    f1V=np.vectorize(f0)
    ff1=f1V(np.transpose([V1]),V2)
    def sqrt_f0_times_u(v1,v2):
        racine_f0=np.exp(-v1*v1/4)*np.exp(-v2*v2/4)
        tau_p=np.exp(-np.sqrt(v1*v1+v2*v2))
        u_n=np.exp(-2*np.pi*n*1j*v2/omega_c)
        return racine_f0*tau_p*u_n
    
    f2V=np.vectorize(sqrt_f0_times_u)
    ff2=f2V(np.transpose([V1]),V2)
    
    
    ffx=[ff1+epsilon1*np.exp(2*np.pi*n*1j*x)*ff2 for x in X]
    return ffx

def fnon_mode_propre(epsilon,k,omega_c):
    def f0(v1,v2):
        return 1/(2*np.pi)*(np.exp(-v1*v1/2)*np.exp(-v2*v2/2))
    f1V=np.vectorize(f0)
    ff1=f1V(np.transpose([V1]),V2)
    def vec_propre(v1,v2):
        racine_f0=np.exp(-v1*v1/4)*np.exp(-v2*v2/4)
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
        return racine_f0*tau_p*u_n*e_m
    
    f2V=np.vectorize(vec_propre)
    ff2=f2V(np.transpose([V1]),V2)
    
    
    ffx=[ff1+epsilon1*np.exp(2*np.pi*n*1j*x)*ff2 for x in X]
    return ffx
 

f=f_ini(epsilon,k)



def Poisson_trap(f):
    f1=copy.copy(f)
    
    #initialisation
    e = np.zeros((N,1))

    #integration en direction v1 et v2, méthodes des trapèzes
    S=[]
    
    for i in range(len(X)):
        #intégration en v1
        s1=np.sum(f1[i][0:M1-1,:],axis=0)+np.sum(f1[i][1:M1,:],axis=0)
        
        s=0
        #intégration en v2
        for j in range (len(s1)-1):
            s+=s1[j]+s1[j+1]
        
        S.append(s)
      
    T=dv1*dv2/4*np.array([S])
    T=np.transpose(T)
    rho=np.ones((N,1))-T
    
    #intégration direction x, méthode des trapèzes
    A=np.tril(np.ones((N-1, N-1)))
    R=rho[0:N-1,:]+rho[1:N,:]
    
     
    e[1:N,:]=np.dot(A,R)*dx/2
    #périodicité
    e[0,0]=e[N-1,0]
    #intégration nulle 
    e=e-sum(e[1:N])/(N-1)
    
    return e

e=Poisson_trap(f)




def SLC(k,f,e,X,V1,V2):
    f1=copy.deepcopy(f)
    k1=copy.deepcopy(k)
    e1=copy.deepcopy(e)
    X1=copy.deepcopy(X)
    V1bis=copy.deepcopy(V1)
    V2bis=copy.deepcopy(V2)
    #initialisation E et norme E
    E=np.zeros((N,Nbt))
    Enum=np.zeros((1,Nbt))
    #première colonne E et première valeur norme E
    E[:,0:1]=e
    Enum[0]=np.linalg.norm(e1)
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
         print("Time=",t*dt,"et Niter=",t)
         #affichage masse initiale
         print("   masse initiale= ",real_m_ini/L)
    
         #affichage masse à chaque instant
         masse_non_norma=0
         for k in range(N-1):
          for i in range(M1):
           for j in range(M2):
              masse_non_norma+=f1[k][i,j]
         masse=masse_non_norma*dx*dv1*dv2
         print("   masse courante= ",masse/L)
         print("   Erreur relative de masse= ", (masse-real_m_ini)/real_m_ini)
         
         masse_quadra=0
         for k in range(N-1):
          for i in range(M1-1):
           for j in range(M2-1):
              masse_quadra+=abs(f1[k][i,j])**2
         masse_quadra=masse_quadra*dx*dv1*dv2
         
         masse_quadra_E=0
         for k in range(N-1):
             masse_quadra_E+=abs(E[k,t-1:t])**2
             masse_quadra_E=masse_quadra_E*dx
         print("   énergie quadratique phase= ",masse_quadra+masse_quadra_E)
         """
         mf0=[np.abs(np.max(f0[i])) for i in range(len(f1))]
         mff0=max(mf0)
         
         f_diff=[f1[i]-f0[i] for i in range(len(f0))]
         
         mf=[max(np.abs(np.max(f_diff[i])),np.abs(np.min(f_diff[i]))) for i in range(len(f1))]
         mff=max(mf)
         l.append(mff)
         mE=linalg.norm(E[:,t-1:t],np.inf)
         print("   Max Densité f-f0= ",mff/mff0)
         print("   Max Champ électrique E= ",mE)
         p=randrange(M1)
         print("   p=",p)
         
         fig = plt.figure(t)
         fig = plt.figure(figsize= (15, 10))
         plt.plot(V2,[f0[0][p,j] for j in range(M2)],label="f initial")
         plt.plot(V2,[f1[0][p,j] for j in range(M2)],'x', label="f au temps "+str(t*dt)+" selon la direction "+str(p))
         
         plt.xlabel('v2')
         plt.ylabel('f selon v2')
         plt.legend()
         """
         
        
       if (t % modbis==0):
           Time=np.array([i*dt for i in range(t)])
           Enumt=Enum[0,0:t]
           Ethet=4*epsilon*0.424666*np.exp(-0.0661*Time)*np.linalg.norm(np.sin(0.4*Xbis))*np.cos(1.2850*Time-0.3357725)
          
           plt.figure(figsize = (30, 20))
           plt.plot(   Time,   np.log(Enumt), color='black', label=r'$E$ num')
           axes = plt.gca()
           axes.xaxis.set_ticks([i*t*dt/10 for i in range(11)])
           axes.xaxis.set_tick_params(length=5, width=3,labelsize=20)
           axes.yaxis.set_tick_params(length=5, width=3,labelsize=20)
           if flag==2:
               plt.plot(   Time,   np.log(np.abs(Ethet)),  color='red', label=r"$E$ theo")
               plt.title(r'Landau Damping with $k=$'+str(k1),fontsize = 40)
               N_t=math.floor(t*dt/(L/dv1))
               for i in range(1,N_t+1):
                   plt.axvline(x=i*L/dv1,color='green')
           if flag==1:
               plt.plot(   Time,   np.log(np.abs(Ethet)),  color='red', label=r"$E$ theo")
               plt.title(r'Landau-Bernstein paradox with $k=$'+str(k1)+" et $\omega_c=$ "+str(omega_c),fontsize = 60)
               
               N_t=math.floor(t*dt/(2*np.pi/omega_c))
               for i in range(1,N_t+1):
                   plt.axvline(x=i*2*np.pi/omega_c,color='green')
               
           
           if flag==0:
                plt.title(r"Two-stream instability with $k= $"+str(k1)+" et $v0=$ "+str(v0),fontsize = 40)
  
           if flag==-1:
               plt.plot(   Time,   np.log(np.abs(Ethet)),  color='red', label=r"$E$ theo $k=$"+str(k))  
               plt.title(r"Two-stream initial condition with $k= $"+str(k1)+", $v0=$ "+str(v0)+" et $\omega_c=$ "+str(omega_c),fontsize = 40)
           
           plt.ylabel('Norm of the electric field',fontsize=50)
           plt.xlabel('Time',fontsize=50)
           plt.legend(loc = 'upper right',prop={'size':60})
           
           
           LocalDestinationPath = '/home/sidd/Bureau/Recherche/Landau Bernstein Scattering/Version 5' # Changer here for your  local directory
           os.chdir(LocalDestinationPath)
           plt.savefig("Norme du Champ électrique jusqu'à l'iter "+str(t))
           
           
       """ 
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
       
         
       masse_quadra=0
       for k in range(N-1):
          for i in range(M1-1):
           for j in range(M2-1):
              masse_quadra+=abs(f1[k][i,j])**2
       masse_quadra=masse_quadra*dx*dv1*dv2
         
       masse_quadra_E=0
       for k in range(N-1):
             masse_quadra_E+=abs(E[k,t-1:t])**2
       masse_quadra_E=masse_quadra_E*dx
       print("   énergie quadratique phase 1= ",masse_quadra+masse_quadra_E)  
         
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
       #résolution de Poisson
       E[:,t:t+1]= Poisson_trap(f1)
#        print ('E=',Poisson_trap(dx,dv,f1))
       if (t % mod == 0):
         I=0.
         for j in range(N-1):
          I=I+E[:,t:t+1][j]
         
         #print ('   Integrale E= ', I*dx)
       Enum[0,t]=np.linalg.norm(E[0:N-1,t:t+1])
       
       
       #résolution à x et v2 constants  
       for k in range(N):
           for j in range(M2):
               V1bisp=V1bis+(E[k,t]+omega_c*V2bis[j])*dt
               
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
       
       masse_quadra=0
       for k in range(N-1):
          for i in range(M1-1):
           for j in range(M2-1):
              masse_quadra+=abs(f1[k][i,j])**2
       masse_quadra=masse_quadra*dx*dv1*dv2
         
       masse_quadra_E=0
       for k in range(N-1):
             masse_quadra_E+=abs(E[k,t-1:t])**2
       masse_quadra_E=masse_quadra_E*dx
       print("   énergie quadratique phase 2= ",masse_quadra+masse_quadra_E)
       
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
               #a=scipy.interpolate.CubicSpline(V2bis,fki,bc_type='clamped')(V2bisp)
               a=scipy.interpolate.CubicSpline(V2bis,fki,bc_type='periodic')(V2bisp)
               f1[k][i:i+1,:]=np.array([[a[j] for j in range(M2)]])
       
       masse_quadra=0
       for k in range(N-1):
          for i in range(M1-1):
           for j in range(M2-1):
              masse_quadra+=abs(f1[k][i,j])**2
       masse_quadra=masse_quadra*dx*dv1*dv2
         
       masse_quadra_E=0
       for k in range(N-1):
             masse_quadra_E+=abs(E[k,t-1:t])**2
       masse_quadra_E=masse_quadra_E*dx
       print("   énergie quadratique phase 3= ",masse_quadra+masse_quadra_E)  
         
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
    return [E,Enum,f1]

start = time.time()    

[E,Enum,fend]=SLC(k,f,e,X,V1,V2)

end = time.time()
print("Elapsed (after compilation) = %s" % (end - start))


#Landau théorique pour k=0.4
Ethe=4*epsilon*0.424666*np.exp(-0.0661*Time)*np.linalg.norm(np.sin(0.4*Xbis))*np.cos(1.2850*Time-0.3357725)

#Landau théorique pour k=0.5
Ethe1=4*epsilon*0.3677*np.exp(-0.1533*Time)*np.linalg.norm(np.sin(0.5*X))*np.cos(1.4156*Time-0.536245)
#visualisation

fig = plt.figure(1)
fig = plt.figure(figsize = (30, 20))
plt.plot(   Time,   np.log(np.transpose(Enum)), color='black', label=r'$E$ num')

#opérations sur les axes
axes = plt.gca()

if flag==2:
    N_T=math.floor(T/(L/dv1))
    for i in range(1,N_T+1):
        plt.axvline(x=i*L/dv1,color='green',linestyle='--')
    plt.plot(   Time,   np.log(np.abs(Ethe)),  color='red', label=r"$E$ theo")
    plt.title(r'Landau Damping with $k=$'+str(k),fontsize = 40)
  
if flag==1:
  plt.plot(   Time,   np.log(np.abs(Ethe)),  color='red', label=r"$E$ theo")
  
  N_T=math.floor(T/(2*np.pi/omega_c))
  for i in range(1,N_T+1):
      plt.axvline(x=i*2*np.pi/omega_c,color='green')
      
  plt.xticks([i*2*np.pi/omega_c for i in range(1,4)]+[i*T/10 for i in range(11)],\
            [r'$\frac{2\pi}{\omega_c}$']+[str(i)+r'$\times\frac{2\pi}{\omega_c}$' for i in range(2,4)]+[str(i*20) for i in range(11)])
  plt.gca().get_xticklabels()[0].set_color("green")
  plt.gca().get_xticklabels()[1].set_color("green")
  plt.gca().get_xticklabels()[2].set_color("green")
  
  """ 
  N_T=math.floor(T/(L/dv1))
  for i in range(1,N_T+1):
        plt.axvline(x=i*L/dv1,color='green',linestyle='--')
  """
  plt.title(r'Bernstein-Landau paradox with $k=$'+str(k)+" et $\omega_c=$ "+str(omega_c),fontsize = 60)
  
if flag==0:
  plt.title(r"Two-stream instability with $k= $"+str(k)+" et $v0=$ "+str(v0),fontsize = 40)
  
if flag==-1:
  plt.plot(   Time,   np.log(np.abs(Ethe)),  color='red', label=r"$E$ Theo $k=$"+str(k))  
  plt.title(r"Two-stream initial condition with $k= $"+str(k)+", $v0=$ "+str(v0)+" et $\omega_c=$ "+str(omega_c),fontsize = 40)

#axes.xaxis.set_ticks([i*T/10 for i in range(11)])
axes.xaxis.set_tick_params(length=5, width=3,labelsize=20)
axes.yaxis.set_tick_params(length=5, width=3,labelsize=20)
plt.ylabel('Norm of the electric field',fontsize=50)
plt.xlabel('Time',fontsize=50)
plt.legend(loc = 'upper right',prop={'size':60})

LocalDestinationPath = '/home/sidd/Bureau/Recherche/Landau Bernstein Scattering/Version 5' 
os.chdir(LocalDestinationPath)
plt.savefig("Norme du Champ électrique jusqu'à l'iter "+str(Nbt))
"""
"""
    
    
    


