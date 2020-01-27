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
from scipy.interpolate import RectBivariateSpline
from scipy.interpolate import interp2d
from scipy import interpolate


from mode_propre_lambda import dichotomie
from mode_propre_lambda import f_mode_propre_nm


#fréquence cyclotronique
omega_c=0

#mode de Fourier
n=1

flag=1

"""   
m=2
c=1.001
d=1.4

start=time.time()
lambda_m=dichotomie(c,d,omega_c,n)
print("lambda_"+str(m)+" = ",lambda_m)          
end=time.time()

print("Elapsed time for dichotomy = ",(end-start))
"""

#position
N=33
k=0.4
L=2*np.pi/k
dx=L/(N-1)
X=np.linspace(0,L,N,endpoint=True)
Xbis=np.linspace(0,L,N-1,endpoint=False)
#print(X,Xbis)

#vitesse 1
M1=33
A1=5
dv1=2*A1/(M1-1)
V1=np.linspace(-A1,A1,M1,endpoint=True)


#vitesse 2
M2=33
A2=5
dv2=2*A2/(M2-1)
V2=np.linspace(-A2,A2,M2,endpoint=True)

#temps
T=200
mod=1
modbis=500
dt=0.01
Nbt=int(T/dt)
Time=np.array([i*dt for i in range(Nbt)])



print ('--------------')
print ('dx =',dx)
print ('dv1=',dv1)
print ('dv2=',dv2)
print ('dt=',dt)
print ('Flag (donnée initiale)= ',flag)
print ('--------------')


"""
def f_mode_propre(omega_c):
    def u(v1,v2):
        tau_p=np.exp(-(v1**2+v2**2)/4)*((v1**2+v2**2)/2-1)
        u_n=np.exp(-n*1j*v2/omega_c)
        return tau_p*u_n
    
    f2V=np.vectorize(u)
    ff2=f2V(np.transpose([V1]),V2)
    
    
    ffx=[np.exp(n*1j*x)*ff2 for x in X]
    return ffx

def f_mode_propre1(omega_c):
    def u(v1,v2):
        return (np.exp(-v1*v1/2)*np.exp(-v2*v2/2))*((v1**2+v2**2)/2-1)
    
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
"""

def f_rec(k):
    def u(v1,v2):
        tau_p=np.exp(-(v1**2+v2**2)/4)
        return tau_p
    
    f2V=np.vectorize(u,otypes=[complex])
    ff2=f2V(np.transpose([V1]),V2)
    
    
    ffx=[np.cos(k*x)*ff2 for x in X]
    return ffx
 
f=f_rec(k)
print(f[0].shape)

"""
start=time.time()
f=f_mode_propre_nm(V1,V2,X,omega_c,n,lambda_m)
#print(vec_propre)
end=time.time()

print("Elapsed time for loading init = ",(end-start))
"""


def mult_Poisson(f):
	f1=copy.deepcopy(f)
	for k in range(N):
		for i in range(M1):
			for j in range(M2):
				f1[k][i,j]=f1[k][i,j]*np.exp(-(V1[i]**2+V2[j]**2)/4)
	return f1

def Poisson_trap(f):
    f0=copy.deepcopy(f)
    f1=mult_Poisson(f0)
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
    rho=-T
    
    plt.plot(X,rho)
    mC=max(np.absolute(rho))
    #print("   Max charge= ",mC)
    
    #intégration direction x, méthode des trapèzes
    A=np.tril(np.ones((N-1, N-1)))
    R=rho[0:N-1,:]+rho[1:N,:]
    
     
    e[1:N,:]=np.dot(A,R)*dx/2
    
    #périodicité
    e[0,0]=e[N-1,0]
    #intégration nulle 
    e=e-sum(e[1:N])/(N-1)
    #print(e)
    return e

einit=Poisson_trap(f)
   

ebis=[-2*np.pi/k*np.sin(k*x) for x in X]

plt.plot(X,einit)
plt.plot(X,ebis,'x')
"""
"""

def mult_Ampere(f):
	f1=copy.deepcopy(f)
	for k in range(N):
		for i in range(M1):
			for j in range(M2):
				f1[k][i,j]=f1[k][i,j]*V1[i]*np.exp(-(V1[i]**2+V2[j]**2)/4)
	return f1


def Ampere_trap(f):
    f0=copy.deepcopy(f)
    f1=mult_Ampere(f0)
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
    rho=T
    mC=max(np.absolute(rho))
    #print("   Max charge= ",mC)
    """
    #intégration direction x, méthode des trapèzes
    A=np.tril(np.ones((N-1, N-1)))
    R=rho[0:N-1,:]+rho[1:N,:]
    
     
    e[1:N,:]=np.dot(A,R)*dx/2
    """
    #périodicitéhttps://stackoverflow.com/questions/45960590/how-to-add-python-console-in-spyder
    rho[0,0]=rho[N-1,0]
    
    #print(rho)
    #intégration nulle 
    rho=rho-sum(rho[1:N])/(N-1)
    #print(sum(rho[1:N]))
    return rho


e=Ampere_trap(f)


def second_membre(e):
    def g(v1,v2):
        return v1*np.exp(-(v1**2+v2**2)/4)
    gV=np.vectorize(g)
    ggV=gV(np.transpose([V1]),V2)
    
    
    ggx=[e[k]*ggV for k in range(len(e))]
    return ggx

#einit=np.transpose(np.array([-n*1j*np.exp(n*1j*X)]))

#print(einit.shape)

def SLC(k,f,einit,X,V1,V2):
    f0=copy.deepcopy(f)
    f1=copy.deepcopy(f)
    k1=copy.deepcopy(k)
    e1=copy.deepcopy(einit)
    X1=copy.deepcopy(X)
    V1bis=copy.deepcopy(V1)
    V2bis=copy.deepcopy(V2)
    #initialisation E et norme E
    E=np.zeros((N,Nbt),dtype=complex)
    Enum=np.zeros((1,Nbt),dtype=complex)
    #première colonne E et première valeur norme E
    E[:,0:1]=e1
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
       Temps=t*dt
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
         
         #affichage énergie quadratique
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
         print("   énergie u= ",masse_quadra)
         print("   énergie F= ",masse_quadra_E)
         print("   énergie quadratique= ",masse_quadra+masse_quadra_E)
         
         mf0=[np.abs(np.max(f0[i])) for i in range(len(f1))]
         mff0=max(mf0)
         
         f_diff=[f1[i]-f0[i] for i in range(len(f0))]
         
         mf=[max(np.abs(np.max(f_diff[i])),np.abs(np.min(f_diff[i]))) for i in range(len(f1))]
         #mf=[np.linalg.norm(f_diff[i]) for i in range(len(f1))]

         mff=max(mf)
         l.append(mff)
         mE=linalg.norm(E[:,t-1:t],np.inf)
         print("   Max Densité f-f0= ",mff)
         print("   Max Champ électrique E= ",mE)
         
        
       if (t % modbis==0):
           Time=np.array([i*dt for i in range(t)])
           Enumt=Enum[0,0:t]
           #Ethet=4*0.424666*np.exp(-0.0661*Time)*np.linalg.norm(np.sin(0.4*Xbis))*np.cos(1.2850*Time-0.3357725)
           plt.figure(figsize = (30, 20))
           plt.plot(   Time,   np.log(Enumt), color='black', label=r'$E$ num')
           axes = plt.gca()
           axes.xaxis.set_ticks([i*t*dt/10 for i in range(11)])
           axes.xaxis.set_tick_params(length=5, width=3,labelsize=20)
           axes.yaxis.set_tick_params(length=5, width=3,labelsize=20)
           #if flag==1:
           #plt.plot(   Time,   np.log(np.abs(Ethet)),  color='red', label=r"$E$ theo $k=$"+str(k))
           plt.title(r'Landau-Bernstein paradox with $k=$'+str(k1)+" et $\omega_c=$ "+str(omega_c),fontsize = 40)
               
           #N_t=math.floor(t*dt/(2*np.pi/omega_c))
           #for i in range(1,N_t+1):
           #   plt.axvline(x=i*2*np.pi/omega_c,color='green')
           plt.ylabel('Norm of the electric field',fontsize=30)
           plt.xlabel('Time',fontsize=30)
           plt.legend(loc = 'upper right',prop={'size':20})
           
           
           LocalDestinationPath = '/home/sidd/Bureau/Recherche/Landau Bernstein Scattering/Partie numérique' # Changer here for your  local directory
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
       print("   énergie u phase 1= ",masse_quadra)
       print("   énergie F phase 1= ",masse_quadra_E)
       print("   énergie quadratique phase 1= ",masse_quadra+masse_quadra_E)
       
       #résolution étape terme source+Ampere
       ftemp=copy.copy(f1)
       Etemp=E[:,t-1:t]
       """
       #Euler explicite en u
       for k in range(N):
           f1[k]=f1[k]-dt*second_membre(E[:,t-1:t])[k]
           
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
       
       
       #euler explicite en F
       E[:,t:t+1]=E[:,t-1:t]+dt*Ampere_trap(ftemp)
       
       
       masse_quadra=0
       for k in range(N-1):
          for i in range(M1-1):
           for j in range(M2-1):
              masse_quadra+=abs(f1[k][i,j])**2
       masse_quadra=masse_quadra*dx*dv1*dv2
         
       masse_quadra_E=0
       for k in range(N-1):
             masse_quadra_E+=abs(E[k,t:t+1])**2
       masse_quadra_E=masse_quadra_E*dx
       print("   énergie quadratique phase 2bis= ",masse_quadra+masse_quadra_E)
       
       #prédicteur-correcteur en u et F
       
       #prédicteur
       Etemp=Etemp+dt/2*Ampere_trap(ftemp)
       
       for k in range(N):
           ftemp[k]=ftemp[k]-dt/2*second_membre(E[:,t-1:t])[k]
       
       #correcteur
       E[:,t:t+1]=E[:,t-1:t]+dt*Ampere_trap(ftemp)
       
       for k in range(N):
           f1[k]=f1[k]-dt*second_membre(Etemp)[k] 
       """
       #Crank-Nicolson en u et F
       
       alpha=dt/2
       b_uno=ftemp
       for k in range(N):
         b_uno[k]=b_uno[k]-alpha*second_membre(Etemp)[k] 
         
       b_dos=Etemp+alpha*Ampere_trap(ftemp)
       
       E[:,t:t+1]=b_dos+alpha*Ampere_trap(b_uno)/(1+alpha**2*2*np.pi)
       
       for k in range(N):
           f1[k]=f1[k]-alpha*(second_membre(Etemp)[k]+second_membre(E[:,t:t+1])[k]) 
       
       masse_quadra=0
       for k in range(N-1):
          for i in range(M1-1):
           for j in range(M2-1):
              masse_quadra+=abs(f1[k][i,j])**2
       masse_quadra=masse_quadra*dx*dv1*dv2
         
       masse_quadra_E=0
       for k in range(N-1):
             masse_quadra_E+=abs(E[k,t:t+1])**2
       masse_quadra_E=masse_quadra_E*dx
       print("   énergie u phase 2= ",masse_quadra)
       print("   énergie F phase 2= ",masse_quadra_E)
       print("   énergie quadratique phase 2= ",masse_quadra+masse_quadra_E) 
       
       #résolution à x et v2 constants  
       
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
               
               
               #V1bisp=V1bisp+(V1bisp<V1bis[0])*(V1bis[-1]-V1bis[0])-(V1bisp>V1bis[-1])*(V1bis[-1]-V1bis[0])
               for k in range(N):
                   fkj=[f1[k][i,j] for i in range(M1)]
                   fkj[0]=fkj[-1]
            
                   
                   for i in range(M1):
                       if (V1bisp[i]>=V1[-1] or V1bisp[i]<=V1[0]):
                           fkj[i]=0       
               
                   
               
                   #interpolation par spline cubique
                   #a=scipy.interpolate.CubicSpline(V1bis,fkj,bc_type='clamped')(V1bisp)
                   a=scipy.interpolate.CubicSpline(V1bis,fkj,bc_type='periodic')(V1bisp)               
                   f1[k][:,j:j+1]=np.transpose(np.array([[a[i] for i in range(M1)]]))
       
       masse_non_norma=0
       for k in range(N-1):
          for i in range(M1-1):
           for j in range(M2-1):
              masse_non_norma+=f1[k][i,j]
       masse=masse_non_norma*dx*dv1*dv2
       print("     masse courante phase 2= ",masse/L)
       print("     Erreur relative de masse phase 2= ", (masse-real_m_ini)/real_m_ini)
       
       masse_quadra=0
       for k in range(N-1):
          for i in range(M1-1):
           for j in range(M2-1):
              masse_quadra+=abs(f1[k][i,j])**2
       masse_quadra=masse_quadra*dx*dv1*dv2
         
       masse_quadra_E=0
       for k in range(N-1):
             masse_quadra_E+=abs(E[k,t:t+1])**2
       masse_quadra_E=masse_quadra_E*dx
       print("   énergie quadratique phase 3= ",masse_quadra+masse_quadra_E)
       
       
       #résolution à x et v1 constants    
       
           
           
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
               for k in range(N):
                   fki=[f1[k][i,j] for j in range(M2)]
                   fki[0]=fki[-1]
               
                   
                   #for j in range(M2):
                   #    if (V2bisp[j]>=V2[-1] or V2bisp[j]<=V2[0]):
                   #        fki[j]=0     
                   
               
                   #interpolation par spline cubique
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
             masse_quadra_E+=abs(E[k,t:t+1])**2
       masse_quadra_E=masse_quadra_E*dx
       print("   énergie quadratique phase 4= ",masse_quadra+masse_quadra_E) 
         
       
       masse_non_norma=0
       for k in range(N-1):
          for i in range(M1-1):
           for j in range(M2-1):
              masse_non_norma+=f1[k][i,j]
       masse=masse_non_norma*dx*dv1*dv2
       print("     masse courante phase 3= ",masse/L)
       print("     Erreur relative de masse phase 3= ", (masse-real_m_ini)/real_m_ini)
       
       
       """
       if(t % mod==0):
         fxv3=np.real(f0[n])
         fxv4=np.imag(f0[n])
         fxv5=np.absolute(f0[n])
         
                  
         
         fxv3end=np.real(f1[n])
         fxv4end=np.imag(f1[n])
         fxv5end=np.absolute(f1[n])
         
         
         
         er=np.real(e1)
         ei=np.imag(e1)
         
         
         Erend=np.real(E[:,t:t+1])
         Eiend=np.imag(E[:,t:t+1])
         
         
         
         
         
         plt.figure()
         plt.figure(figsize = (30, 20))
         plt.subplot(2, 3, 1)
         plt.pcolormesh(V2, V1, fxv3,cmap='RdBu')
         plt.title("densité u initiale REELLE dans le plan V1-V2 pour x="+str(0))
         plt.colorbar()
         plt.ylabel('V2')
         plt.xlabel('V1')
         axes = plt.gca()
         axes.xaxis.set_tick_params(length=5, width=3,labelsize=20)
         axes.yaxis.set_tick_params(length=5, width=3,labelsize=20)   
         plt.legend()
         plt.subplot(2, 3, 2)
         plt.pcolormesh(V2, V1, np.cos(lambda_m*t*dt)*fxv3-np.sin(lambda_m*t*dt)*fxv4,cmap='RdBu')
         plt.title("densité u théorique REELLE au temps "+str(t)+" dans le plan V1-V2 pour x="+str(0))
         plt.colorbar()
         plt.ylabel('V2')
         plt.xlabel('V1')
         axes = plt.gca()
         axes.xaxis.set_tick_params(length=5, width=3,labelsize=20)
         axes.yaxis.set_tick_params(length=5, width=3,labelsize=20)   
         plt.legend()
         plt.subplot(2, 3, 3)
         plt.pcolormesh(V2, V1, fxv3end,cmap='RdBu')
         plt.title("densité u numérique REELLE au temps "+str(t)+" dans le plan V1-V2 pour x="+str(0))
         plt.colorbar()
         plt.ylabel('V2')
         plt.xlabel('V1')
         axes = plt.gca()
         axes.xaxis.set_tick_params(length=5, width=3,labelsize=20)
         axes.yaxis.set_tick_params(length=5, width=3,labelsize=20)   
         plt.legend()
         plt.subplot(2, 3, 4)
         plt.pcolormesh(V2, V1, fxv4,cmap='RdBu')
         plt.title("densité u initiale IMAGINAIRE dans le plan V1-V2 pour x="+str(0))
         plt.colorbar()
         plt.ylabel('V2')
         plt.xlabel('V1')
         axes = plt.gca()
         axes.xaxis.set_tick_params(length=5, width=3,labelsize=20)
         axes.yaxis.set_tick_params(length=5, width=3,labelsize=20)   
         plt.legend()
         plt.subplot(2, 3, 5)
         plt.pcolormesh(V2, V1, np.sin(lambda_m*t*dt)*fxv3+np.cos(lambda_m*t*dt)*fxv4,cmap='RdBu')
         plt.title("densité u théorique IMAGINAIRE au temps "+str(t)+" dans le plan V1-V2 pour x="+str(0))
         plt.colorbar()
         plt.ylabel('V2')
         plt.xlabel('V1')
         axes = plt.gca()
         axes.xaxis.set_tick_params(length=5, width=3,labelsize=20)
         axes.yaxis.set_tick_params(length=5, width=3,labelsize=20)   
         plt.legend()
         plt.subplot(2, 3, 6)
         plt.pcolormesh(V2, V1, fxv4end,cmap='RdBu')
         plt.title("densité u numérique IMAGINAIRE au temps "+str(t)+" dans le plan V1-V2 pour x="+str(0))
         plt.colorbar()
         plt.ylabel('V2')
         plt.xlabel('V1')
         axes = plt.gca()
         axes.xaxis.set_tick_params(length=5, width=3,labelsize=20)
         axes.yaxis.set_tick_params(length=5, width=3,labelsize=20)   
         plt.legend()
         LocalDestinationPath = '/home/sidd/Bureau/Programmation/Python/Images NumKin 2019/VAlin' 
         os.chdir(LocalDestinationPath)
         plt.savefig("u_VA_"+str(t))
         
         
         plt.figure()
         plt.figure(figsize = (30, 20))
         plt.subplot(2, 2, 1)
         plt.pcolormesh(V2, V1, fxv5,cmap='RdBu')
         plt.title("densité u initiale MODULE dans le plan V1-V2 pour x="+str(0))
         plt.colorbar()
         plt.ylabel('V2')
         plt.xlabel('V1')
         axes = plt.gca()
         axes.xaxis.set_tick_params(length=5, width=3,labelsize=20)
         axes.yaxis.set_tick_params(length=5, width=3,labelsize=20)   
         plt.legend()
         plt.subplot(2, 2, 2)
         plt.pcolormesh(V2, V1, fxv5end,cmap='RdBu')
         plt.title("densité u numérique MODULE au temps "+str(t)+" dans le plan V1-V2 pour x="+str(0))
         plt.colorbar()
         plt.ylabel('V2')
         plt.xlabel('V1')
         axes = plt.gca()
         axes.xaxis.set_tick_params(length=5, width=3,labelsize=20)
         axes.yaxis.set_tick_params(length=5, width=3,labelsize=20)   
         plt.legend()
         plt.subplot(2, 3, 4)
         plt.plot(X, er,'x',label="initial")
         plt.plot(X,np.cos(lambda_m*t*dt)*er-np.sin(lambda_m*t*dt)*ei,label="théorique au temps"+str(t))
         plt.plot(X, Erend,label="numérique au temps "+str(t))
         plt.title("champ électrique REELLE")
         plt.ylabel('E réel')
         plt.xlabel('X')
         axes = plt.gca()
         plt.xticks([0,np.pi/2, np.pi, 3*np.pi/2,2*np.pi], [r'$0$',r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$', r'$2\pi$'])
         axes.xaxis.set_tick_params(length=5, width=3,labelsize=20)
         axes.yaxis.set_tick_params(length=5, width=3,labelsize=20)   
         plt.legend()
         plt.grid()
         plt.subplot(2, 3, 5)
         plt.plot(X, ei,'x',label="initial")
         plt.plot(X,np.sin(lambda_m*t*dt)*er+np.cos(lambda_m*t*dt)*ei,label="théorique au temps"+str(t))
         plt.plot(X, Eiend,label="numérique au temps "+str(t))
         plt.title("champ électrique IMAGINAIRE")
         plt.ylabel('E imaginaire')
         plt.xlabel('X')
         axes = plt.gca()
         plt.xticks([0,np.pi/2, np.pi, 3*np.pi/2,2*np.pi], [r'$0$',r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$', r'$2\pi$'])
         axes.xaxis.set_tick_params(length=5, width=3,labelsize=20)
         axes.yaxis.set_tick_params(length=5, width=3,labelsize=20)   
         plt.legend()
         plt.grid()
         plt.subplot(2, 3, 6)
         plt.axis([0,2*np.pi,0, 1.2])
         plt.plot(X, np.absolute(einit),'x',label="initial")
         #plt.plot(X,np.absolute(np.sin(lambda_m*T)*er+np.cos(lambda_m*T)*ei),label="théorique final")
         plt.plot(X, np.absolute(E[:,t:t+1]),label="numérique au temps "+str(t))
         plt.title("champ électrique MODULE")
         plt.ylabel('E module')
         plt.xlabel('X')
         axes = plt.gca()
         plt.xticks([0,np.pi/2, np.pi, 3*np.pi/2,2*np.pi], [r'$0$',r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$', r'$2\pi$'])
         axes.xaxis.set_tick_params(length=5, width=3,labelsize=20)
         axes.yaxis.set_tick_params(length=5, width=3,labelsize=20)  
         plt.legend()
         plt.grid()
         LocalDestinationPath = '/home/sidd/Bureau/Programmation/Python/Images NumKin 2019/VAlin' 
         os.chdir(LocalDestinationPath)
         plt.savefig("mu_F_VA_"+str(t))
         """
       end=time.time()
       print("   Elapsed for one iteration = %s" % (end - start))
       
       Enum[0,t]=np.linalg.norm(E[0:N-1,t:t+1])
    return [E,Enum,f1]


start = time.time()    

[E,Enum,fend]=SLC(k,f,einit,X,V1,V2)

end = time.time()
print("Elapsed (after compilation) = %s" % (end - start))




u_end_bis=np.zeros((N,M1))

for n in range(len(X)):
   for i in range(len(V1)):
         u_end_bis[n,i]=np.real(fend[n][i,33])
         
print(u_end_bis.shape)
         


fig1=plt.figure()
plt.figure(figsize = (15, 10))
plt.pcolormesh(V1,X,u_end_bis,cmap='RdBu')
plt.colorbar()

"""
fxv3=np.zeros((M1,M2))
fxv4=np.zeros((M1,M2))
fxv5=np.zeros((M1,M2))
for i in range(M1):
    for j in range(M2):
        fxv3[i,j]=np.real(f[n][i,j])
        fxv4[i,j]=np.imag(f[n][i,j])
        fxv5[i,j]=abs(f[n][i,j])
         

fxv3end=np.zeros((M1,M2))
fxv4end=np.zeros((M1,M2))
fxv5end=np.zeros((M1,M2))
for i in range(M1):
    for j in range(M2):
        fxv3end[i,j]=np.real(fend[n][i,j])
        fxv4end[i,j]=np.imag(fend[n][i,j])
        fxv5end[i,j]=abs(fend[n][i,j])



er=np.zeros((N,1))
ei=np.zeros((N,1))

for k in range(N):
    er[k]=np.real(einit[k])
    ei[k]=np.imag(einit[k])


Erend=np.real(E[:,-1])
Eiend=np.imag(E[:,-1])



plt.figure(figsize = (30, 20))
plt.subplot(2, 3, 1)
plt.pcolormesh(V2, V1, fxv3,cmap='RdBu')
plt.title(r"$t=0$", fontsize=40, y=1.1)
cbar=plt.colorbar()
cbar.ax.tick_params(labelsize=40) 
plt.ylabel('REAL PART',rotation=0, fontsize=40)
#plt.xlabel('V1')
axes = plt.gca()
axes.xaxis.set_tick_params(length=5, width=3,labelsize=40)
axes.yaxis.set_tick_params(length=5, width=3,labelsize=40)
axes.yaxis.set_label_coords(-0.4,0.5)   
#plt.legend()
plt.subplot(2, 3, 2)
plt.pcolormesh(V2, V1, np.cos(lambda_m*T)*fxv3-np.sin(lambda_m*T)*fxv4,cmap='RdBu')
plt.title(r"Theoretical at $t=T_f$", fontsize=40, y=1.1)
cbar=plt.colorbar()
cbar.ax.tick_params(labelsize=40) 
#plt.ylabel('V2')
#plt.xlabel('V1')
axes = plt.gca()
axes.xaxis.set_tick_params(length=5, width=3,labelsize=40)
axes.yaxis.set_tick_params(length=5, width=3,labelsize=40)   
#plt.legend()
plt.subplot(2, 3, 3)
plt.pcolormesh(V2, V1, fxv3end,cmap='RdBu')
plt.title(r"Numerical at $t=T_f$", fontsize=40, y=1.1)
cbar=plt.colorbar()
cbar.ax.tick_params(labelsize=40) 
#plt.ylabel('V2')
#plt.xlabel('V1')
axes = plt.gca()
axes.xaxis.set_tick_params(length=5, width=3,labelsize=40)
axes.yaxis.set_tick_params(length=5, width=3,labelsize=40)   
#plt.legend()
plt.subplot(2, 3, 4)
plt.pcolormesh(V2, V1, fxv4,cmap='RdBu')
#plt.title("densité f initiale IMAGINAIRE dans le plan V1-V2 pour x="+str(0))
cbar=plt.colorbar()
cbar.ax.tick_params(labelsize=40) 
plt.ylabel('IMAGINARY \n PART',rotation=0, fontsize=40)
#plt.xlabel('V1')
axes = plt.gca()
axes.xaxis.set_tick_params(length=5, width=3,labelsize=40)
axes.yaxis.set_tick_params(length=5, width=3,labelsize=40)
axes.yaxis.set_label_coords(-0.37,0.5)     
#plt.legend()
plt.subplot(2, 3, 5)
plt.pcolormesh(V2, V1, np.sin(lambda_m*T)*fxv3+np.cos(lambda_m*T)*fxv4,cmap='RdBu')
#plt.title("densité f finale théorique IMAGINAIRE dans le plan V1-V2 pour x="+str(0))
cbar=plt.colorbar()
cbar.ax.tick_params(labelsize=40) 
#plt.ylabel('V2')
#plt.xlabel('V1')
axes = plt.gca()
axes.xaxis.set_tick_params(length=5, width=3,labelsize=40)
axes.yaxis.set_tick_params(length=5, width=3,labelsize=40)   
#plt.legend()
plt.subplot(2, 3, 6)
plt.pcolormesh(V2, V1, fxv4end,cmap='RdBu')
#plt.title("densité f finale numérique IMAGINAIRE dans le plan V1-V2 pour x="+str(0))
cbar=plt.colorbar()
cbar.ax.tick_params(labelsize=40)   
#plt.ylabel('V2')
#plt.xlabel('V1')
axes = plt.gca()
axes.xaxis.set_tick_params(length=5, width=3,labelsize=40)
axes.yaxis.set_tick_params(length=5, width=3,labelsize=40)   
#plt.legend()


LocalDestinationPath = '/home/sidd/Bureau/Recherche/Landau Bernstein Scattering/Partie numérique' 
os.chdir(LocalDestinationPath)
plt.savefig("u_VA_Tend")


plt.figure(figsize = (30, 20))
plt.subplot(2, 2, 1)
plt.pcolormesh(V2, V1, fxv5,cmap='RdBu')
plt.title(r"$t=0$", fontsize=40, y=1.1)
cbar=plt.colorbar()
cbar.ax.tick_params(labelsize=40) 
plt.ylabel('MODULUS',rotation=0, fontsize=40)
#plt.xlabel('V1')
axes = plt.gca()
axes.xaxis.set_tick_params(length=5, width=3,labelsize=40)
axes.yaxis.set_tick_params(length=5, width=3,labelsize=40)   
axes.yaxis.set_label_coords(-0.25,0.5) 
#plt.legend()
plt.subplot(2, 2, 2)
plt.pcolormesh(V2, V1, fxv5end,cmap='RdBu')
plt.title(r"$t=T_f$", fontsize=40, y=1.1)
cbar=plt.colorbar()
cbar.ax.tick_params(labelsize=40) 
#plt.ylabel('MODULUS',rotation=0, fontsize=40)
#plt.xlabel('V1')
axes = plt.gca()
axes.xaxis.set_tick_params(length=5, width=3,labelsize=40)
axes.yaxis.set_tick_params(length=5, width=3,labelsize=40)   

plt.subplot(2, 3, 4)
plt.plot(X, er,'x',label=r"t=0")
plt.plot(X,np.cos(lambda_m*T)*er-np.sin(lambda_m*T)*ei,'o',label=r"Theoretical $t=T_f$")
plt.plot(X, Erend,label=r"Numerical $t=T_f$")
#plt.title("REAL PART", fontsize=40, y=1.1)
plt.ylabel('ELECTRIC \n FIELD',rotation=0, fontsize=40)
plt.xlabel('REAL \n PART', fontsize=40)
axes = plt.gca()
plt.xticks([0,np.pi/2, np.pi, 3*np.pi/2,2*np.pi], [r'$0$',r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$', r'$2\pi$'])
axes.xaxis.set_tick_params(length=5, width=3,labelsize=40)
axes.yaxis.set_tick_params(length=5, width=3,labelsize=40)   
axes.yaxis.set_label_coords(-0.33,0.5)  
plt.legend(prop={'size': 20})
plt.grid()

plt.subplot(2, 3, 5)
plt.plot(X, ei,'x',label=r"t=0")
plt.plot(X,np.sin(lambda_m*T)*er+np.cos(lambda_m*T)*ei,'o',label=r"Theoretical $t=T_f$")
plt.plot(X, Eiend,label=r"Numerical $t=T_f$")
#plt.title("IMAGINARY PART", fontsize=40, y=1.1)
#plt.ylabel('E imaginaire')
plt.xlabel('IMAGINARY \n PART', fontsize=40)
axes = plt.gca()
plt.xticks([0,np.pi/2, np.pi, 3*np.pi/2,2*np.pi], [r'$0$',r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$', r'$2\pi$'])
axes.xaxis.set_tick_params(length=5, width=3,labelsize=40)
axes.yaxis.set_tick_params(length=5, width=3,labelsize=40)  
plt.legend(prop={'size': 20})
plt.grid()

plt.subplot(2, 3, 6)
plt.axis([0,2*np.pi,0, 1.2])
plt.plot(X, np.absolute(einit),'x',label=r"t=0")
#plt.plot(X,np.absolute(np.sin(lambda_m*T)*er+np.cos(lambda_m*T)*ei),label="théorique final")
plt.plot(X, np.absolute(E[:,-1]),'o',label=r"Numerical $t=T_f$")
#plt.title("MODULE", fontsize=40, y=1.1)
#plt.ylabel('E module')
plt.xlabel('MODULUS', fontsize=40)
axes = plt.gca()
plt.xticks([0,np.pi/2, np.pi, 3*np.pi/2,2*np.pi], [r'$0$',r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$', r'$2\pi$'])
axes.xaxis.set_tick_params(length=5, width=3,labelsize=40)
axes.yaxis.set_tick_params(length=5, width=3,labelsize=40)  
plt.legend(prop={'size': 20})
plt.grid()
LocalDestinationPath = '/home/sidd/Bureau/Recherche/Landau Bernstein Scattering/Partie numérique' 
os.chdir(LocalDestinationPath)
plt.savefig("mu_F_VA_Tend")
"""

