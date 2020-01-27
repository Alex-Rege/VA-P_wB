#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 14:04:05 2019

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
import os

import numba
from numba import jit
from numba import njit
import time

from scipy.interpolate import CubicSpline

from mode_propre_lambda import dichotomie
from mode_propre_lambda import f_mode_propre_nm
from mode_propre_lambda import f_mode_propre_nm_nonlin
#from mode_propre_lambda import f_mode_propre_nm_nonlin_imag


"""
flag=2 Landau Damping
flag=1 Landau Damping avec B
flag=0 Double faisceaux
flag=-1 Double faisceaux avec B
"""

#perturbation
epsilon= 0.1

#fréquence cyclotronique
omega_c=0.5

#mode de Fourier
n=1
#n=randrange(1,100)

#choix de m pour les vecteurs propres
#m=1
#m=randrange(1,100)
    
m=2
c=1.001
d=1.4

start=time.time()
lambda_m=dichotomie(c,d,omega_c,n)
print("lambda_"+str(m)+" = ",lambda_m)          
end=time.time()

print("Elapsed time for dichotomy = ",(end-start))

#position
N=33
k=0.4
L=2*np.pi
dx=L/(N-1)
X=np.linspace(0,L,N,endpoint=True)
Xbis=np.linspace(0,L,N-1,endpoint=False)

#vitesse 1
M1=63
A1=5
dv1=2*A1/(M1-1)
V1=np.linspace(-A1,A1,M1,endpoint=True)


#vitesse 2
M2=63
A2=5
dv2=2*A2/(M2-1)
V2=np.linspace(-A2,A2,M2,endpoint=True)

#temps
T=np.pi/(2*lambda_m)
mod=1
modbis=100
dt=T/20
Nbt=int(T/dt)
Time=np.array([i*dt for i in range(Nbt)])



print ('--------------')
print ('dx =',dx)
print ('dv1=',dv1)
print ('dv2=',dv2)
print ('dt=',dt)
print ('--------------')



def f_mode_propre(epsilon1,omega_c):
    def f0(v1,v2):
        return (np.exp(-v1*v1/2)*np.exp(-v2*v2/2))
    f1V=np.vectorize(f0)
    ff1=f1V(np.transpose([V1]),V2)
    def sqrt_f0_times_u(v1,v2):
        racine_f0=np.exp(-v1*v1/4)*np.exp(-v2*v2/4)
        tau_p=np.exp(-(v1**2+v2**2)/4)*((v1**2+v2**2)/2-1)
        u_n=np.exp(-n*1j*v2/omega_c)
        return racine_f0*tau_p*u_n
    
    f2V=np.vectorize(sqrt_f0_times_u)
    ff2=f2V(np.transpose([V1]),V2)
    
    
    ffx=[ff1+epsilon*np.exp(n*1j*x)*ff2 for x in X]
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
    
    
    ffx=[ff1+epsilon*np.exp(2*np.pi*n*1j*x)*ff2 for x in X]
    return ffx

def f0(V1,V2):
    def f_0(v1,v2):
        r2=v1**2+v2**2
        return np.exp(-r2/2)
    
    f0V=np.vectorize(f_0)
    ff0=f0V(np.transpose([V1]),V2)
    
    ffx=[ff0 for x in X]
    return ffx

f0=f0(V1,V2)



fini_complex=f_mode_propre_nm_nonlin(V1,V2,X,omega_c,n,lambda_m,epsilon)
fini_real=np.real(fini_complex)
fini_imag=np.imag(fini_complex)

fini=[0 for x in X]

for k in range(N):
    fini[k]=fini_real[k]+f0[k]


    
    

def Poisson_trap(f):
    f1=copy.copy(f)
    
    #initialisation
    e = np.zeros((N,1),dtype=complex)

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
    rho=2*np.pi*np.ones((N,1))-T
    
    #intégration direction x, méthode des trapèzes
    A=np.tril(np.ones((N-1, N-1)))
    R=rho[0:N-1,:]+rho[1:N,:]
    
     
    e[1:N,:]=np.dot(A,R)*dx/2
    #périodicité
    e[0,0]=e[N-1,0]
    #intégration nulle 
    e=e-sum(e[1:N])/(N-1)
    
    return e

e=Poisson_trap(fini)

"""
plt.plot(X,(1/epsilon)*np.real(e))
plt.plot(X,(1/epsilon)*np.imag(e))
axes = plt.gca()
plt.xticks([0,np.pi/2, np.pi, 3*np.pi/2,2*np.pi], [r'$0$',r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$', r'$2\pi$'])
axes.xaxis.set_tick_params(length=5, width=3,labelsize=20)
axes.yaxis.set_tick_params(length=5, width=3,labelsize=20)   
plt.legend()
plt.grid()
plt.show()
"""

def SLC(k,f,e,X,V1,V2):
    f0=copy.deepcopy(f)
    f1=copy.deepcopy(f)
    #k1=copy.deepcopy(k)
    e1=copy.deepcopy(e)
    X1=copy.deepcopy(X)
    V1bis=copy.deepcopy(V1)
    V2bis=copy.deepcopy(V2)
    #initialisation E et norme E
    E=np.zeros((N,Nbt),dtype=complex)
    Enum=np.zeros((1,Nbt),dtype=complex)
    
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
          for i in range(M1-1):
           for j in range(M2-1):
              masse_non_norma+=f1[k][i,j]
         masse=masse_non_norma*dx*dv1*dv2
         print("   masse courante= ",masse/L)
         print("   Erreur relative de masse= ", (masse-real_m_ini)/real_m_ini)
         
         
         mf0=[np.abs(np.max(f0[i])) for i in range(len(f1))]
         mff0=max(mf0)
         
         f_diff=[f1[i]-f0[i] for i in range(len(f0))]
         
         mf=[max(np.abs(np.max(f_diff[i])),np.abs(np.min(f_diff[i]))) for i in range(len(f1))]
         mff=max(mf)
         l.append(mff)
         mE=linalg.norm(E[:,t-1:t],np.inf)
         print("   Max Densité f-f0= ",mff/mff0)
         print("   Max Champ électrique E= ",mE)
         
         """
         p=int(M1/2)
         print("   p=",p)
         
         fig = plt.figure(t)
         fig = plt.figure(figsize= (15, 10))
         plt.plot(V2,[f0[0][p,j] for j in range(M2)],label="f initial")
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

[E,Enum,fend1]=SLC(k,fini,e,X,V1,V2)

end = time.time()
print("Elapsed (after compilation) = %s" % (end - start))

ftemp=[0 for x in X]
fendtemp=[0 for x in X]
"""
for k in range(N):
    f[k]=(1/epsilon)*(fini[k]-f0[k])
    fend[k]=(1/epsilon)*(fend1[k]-f0[k])
"""    

def mult_sqrtf0(f):
	f1=copy.deepcopy(f)
	for k in range(N):
		for i in range(M1):
			for j in range(M2):
				f1[k][i,j]=f1[k][i,j]*np.exp((V1[i]**2+V2[j]**2)/4)
	return f1
   
for k in range(N):
    ftemp[k]=fini[k]-f0[k]
    fendtemp[k]=fend1[k]-f0[k]

f=mult_sqrtf0(ftemp)
fini_real_bis=mult_sqrtf0(fini_real)
fini_imag_bis=mult_sqrtf0(fini_imag)
fend=mult_sqrtf0(fendtemp)
"""
#trois figure suivante pour regarder coment le schéma conserve f0  
fig3st=plt.figure()
#te=t*dt
plt.figure(figsize = (15, 10))
#plt.axis([0,16,-10, 10])
plt.pcolormesh(V2, V1, f0[0],cmap='RdBu')
plt.title("densité f initiale REELLE dans le plan V1-V2 pour x="+str(0))
plt.colorbar()
plt.ylabel('V2')
plt.xlabel('V1')
axes = plt.gca()
axes.xaxis.set_tick_params(length=5, width=3,labelsize=20)
axes.yaxis.set_tick_params(length=5, width=3,labelsize=20)   
plt.legend()
plt.show()


fig3st1=plt.figure()
#te=t*dt
plt.figure(figsize = (15, 10))
#plt.axis([0,16,-10, 10])
plt.pcolormesh(V2, V1, fend1[0],cmap='RdBu')
plt.title("densité f initiale REELLE dans le plan V1-V2 pour x="+str(0))
plt.colorbar()
plt.ylabel('V2')
plt.xlabel('V1')
axes = plt.gca()
axes.xaxis.set_tick_params(length=5, width=3,labelsize=20)
axes.yaxis.set_tick_params(length=5, width=3,labelsize=20)   
plt.legend()
plt.show()

fig3st2=plt.figure()
#te=t*dt
plt.figure(figsize = (15, 10))
#plt.axis([0,16,-10, 10])
plt.pcolormesh(V2, V1, fend[0],cmap='RdBu')
plt.title("densité f initiale REELLE dans le plan V1-V2 pour x="+str(0))
plt.colorbar()
plt.ylabel('V2')
plt.xlabel('V1')
axes = plt.gca()
axes.xaxis.set_tick_params(length=5, width=3,labelsize=20)
axes.yaxis.set_tick_params(length=5, width=3,labelsize=20)   
plt.legend()
plt.show()
"""


"""
fig3st=plt.figure()
#te=t*dt
plt.figure(figsize = (15, 10))
#plt.axis([0,16,-10, 10])
plt.pcolormesh(V2, V1, (1/epsilon)*(fini_real[0]),cmap='RdBu')
plt.title("densité f initiale théorique REELLE dans le plan V1-V2 pour x="+str(0))
plt.colorbar()
plt.ylabel('V2')
plt.xlabel('V1')
axes = plt.gca()
axes.xaxis.set_tick_params(length=5, width=3,labelsize=20)
axes.yaxis.set_tick_params(length=5, width=3,labelsize=20)   
plt.legend()
plt.show()

fig3=plt.figure()
#te=t*dt
plt.figure(figsize = (15, 10))
#plt.axis([0,16,-10, 10])
plt.pcolormesh(V2, V1, (1/epsilon)*(np.cos(lambda_m*T)*fini_real[0]-np.sin(lambda_m*T)*fini_imag[0]),cmap='RdBu')
plt.title("densité f finale théorique REELLE dans le plan V1-V2 pour x="+str(0))
plt.colorbar()
plt.ylabel('V2')
plt.xlabel('V1')
axes = plt.gca()
axes.xaxis.set_tick_params(length=5, width=3,labelsize=20)
axes.yaxis.set_tick_params(length=5, width=3,labelsize=20)   
plt.legend()
plt.show()

fig3end=plt.figure()
#te=t*dt
plt.figure(figsize = (15, 10))
#plt.axis([0,16,-10, 10])
plt.pcolormesh(V2, V1, fend[0],cmap='RdBu')
plt.title("densité f finale numérique REELLE dans le plan V1-V2 pour x="+str(0))
plt.colorbar()
plt.ylabel('V2')
plt.xlabel('V1')
axes = plt.gca()
axes.xaxis.set_tick_params(length=5, width=3,labelsize=20)
axes.yaxis.set_tick_params(length=5, width=3,labelsize=20)   
plt.legend()
plt.show()
"""

Erend=E[:,-1]

""" 
figE1=plt.figure()
#te=t*dt
plt.figure(figsize = (15, 10))
#plt.axis([0,16,-10, 10])
plt.plot(X, (1/epsilon)*e,'x',label="initial")
#plt.plot(X,np.cos(lambda_m*T)*er-np.sin(lambda_m*T)*ei,label="théorique final")
plt.plot(X, (1/epsilon)*Erend,label="numérique final")
plt.title("champ électrique REELLE")
plt.ylabel('E réel')
plt.xlabel('X')
axes = plt.gca()
plt.xticks([0,np.pi/2, np.pi, 3*np.pi/2,2*np.pi], [r'$0$',r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$', r'$2\pi$'])
axes.xaxis.set_tick_params(length=5, width=3,labelsize=20)
axes.yaxis.set_tick_params(length=5, width=3,labelsize=20)   
plt.legend()
plt.grid()
plt.show()
"""  
  
plt.figure(figsize = (30, 20))
plt.subplot(2, 3, 1)
plt.pcolormesh(V2, V1, (1/epsilon)*(f[0]),cmap='RdBu') 
plt.title(r"$t=0$", fontsize=40, y=1.1)
cbar=plt.colorbar()
cbar.ax.tick_params(labelsize=40) 
plt.ylabel(r'$\frac{f^n-f_0}{\varepsilon \sqrt{f_0}}$',rotation=0, fontsize=40)
#plt.xlabel('V1')
axes = plt.gca()
axes.xaxis.set_tick_params(length=5, width=3,labelsize=40)
axes.yaxis.set_tick_params(length=5, width=3,labelsize=40)
axes.yaxis.set_label_coords(-0.4,0.5)   


plt.subplot(2, 3, 2)
plt.pcolormesh(V2, V1, (1/epsilon)*(np.cos(lambda_m*T)*fini_real_bis[0]-np.sin(lambda_m*T)*fini_imag_bis[0]),cmap='RdBu')
plt.title(r"Theoretical at $t=T_f$", fontsize=40, y=1.1)
cbar=plt.colorbar()
cbar.ax.tick_params(labelsize=40) 
#plt.ylabel('V2')
#plt.xlabel('V1')
axes = plt.gca()
axes.xaxis.set_tick_params(length=5, width=3,labelsize=40)
axes.yaxis.set_tick_params(length=5, width=3,labelsize=40) 
  
plt.subplot(2, 3, 3)
plt.pcolormesh(V2, V1, (1/epsilon)*fend[0],cmap='RdBu')
plt.title(r"Numerical at $t=T_f$", fontsize=40, y=1.1)
cbar=plt.colorbar()
cbar.ax.tick_params(labelsize=40) 
#plt.ylabel('V2')
#plt.xlabel('V1')
axes = plt.gca()
axes.xaxis.set_tick_params(length=5, width=3,labelsize=40)
axes.yaxis.set_tick_params(length=5, width=3,labelsize=40) 

plt.subplot(2,3,5) 
plt.plot(X, (1/epsilon)*e,'x',label=r"t=0")
plt.plot(X,[np.cos(x) for x in X],'o',label=r"Theoretical $t=T_f$")
plt.plot(X, (1/epsilon)*Erend,label=r"Numerical $t=T_f$")
plt.ylabel(r'$\frac{E^n}{\varepsilon}$',rotation=0, fontsize=40)
#plt.xlabel('REAL \n PART', fontsize=40)
axes = plt.gca()
plt.xticks([0,np.pi/2, np.pi, 3*np.pi/2,2*np.pi], [r'$0$',r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$', r'$2\pi$'])
axes.xaxis.set_tick_params(length=5, width=3,labelsize=40)
axes.yaxis.set_tick_params(length=5, width=3,labelsize=40)   
axes.yaxis.set_label_coords(-0.4,0.5)  
plt.legend(prop={'size': 20})
plt.grid() 
LocalDestinationPath = '/home/sidd/Bureau/Recherche/Landau Bernstein Scattering/Version 7' 
os.chdir(LocalDestinationPath)
plt.savefig("u_F_VPNL")


