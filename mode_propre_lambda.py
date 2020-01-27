
import math
import numpy as np
import matplotlib.pyplot as plt
import scipy
from numpy import linalg
from random import randrange
from random import *
import copy
import os

import numba
from numba import jit
from numba import njit
import time

from scipy.integrate import quad
from scipy.optimize import fsolve
"""
#mode de Fourier 
n=1

#champ magnétique
omega_c=0.5
"""

"""
#position
N=128
k=0.4
L=2*np.pi
dx=L/(N-1)
X=np.linspace(0,L,N,endpoint=True)
Xbis=np.linspace(0,L,N-1,endpoint=False)
#print(X,Xbis)

#vitesse 1
M1=128
A1=10
dv1=2*A1/(M1-1)
V1=np.linspace(-A1,A1,M1,endpoint=True)


#vitesse 2
M2=128
A2=10
dv2=2*A2/(M2-1)
V2=np.linspace(-A2,A2,M2,endpoint=True)


X0=np.linspace(-20,20,100)
Y0=[scipy.special.jv(0,x) for x in X0]
Y1=[scipy.special.jv(1,x) for x in X0]
Y2=[scipy.special.jv(2,x) for x in X0]


fig1=plt.figure()
plt.figure(figsize = (15, 10))
plt.plot(X0,Y0,X0,Y1,X0,Y2)
plt.grid()
plt.show()
"""
#coeff am
def f_eig1(r,m,omega_c,n):
    return r*np.exp(-(r**2)/2)*scipy.special.jv(m,n*r/omega_c)**2


def a_sec(m,omega_c,n):
    return quad(f_eig1,0,np.inf, args=(m,omega_c,n))



def fonc_sec(lam,omega_c,n):
    epsilon=10**(-15)
    somme=0
    m=1
    p=-1
    Tm=m*omega_c/(m*omega_c+lam)*a_sec(m,omega_c,n)[0]
    Tp=p*omega_c/(p*omega_c+lam)*a_sec(p,omega_c,n)[0]
    while abs(Tm) and abs(Tp)>epsilon:
        Tm=m*omega_c/(m*omega_c+lam)*a_sec(m,omega_c,n)[0]
        Tp=p*omega_c/(p*omega_c+lam)*a_sec(p,omega_c,n)[0]
        somme+=Tm+Tp
        m=m+1
        p=p-1
    #print("(m,p)=",(m,p))
    return -1-2*np.pi/(n**2)*somme


#print(fonc_sec(0,omega_c,n)+1+2*np.pi*(1-a_sec(0,omega_c,n)[0]))
"""
L=6*omega_c
L1=-L
L2=L
N=401
H=1
X=np.linspace(L1,L2,N)


x=2*L*random()-L
start = time.time()    

fonc_sec(x,omega_c,n)

end = time.time()
print("Elapsed (after compilation) = %s" % (end - start))


start=time.time()

Y=[fonc_sec(x,omega_c,n) for x in X]
Z=[x*0 for x in X]
end=time.time()
print("Elapsed (after compilation) = %s" % (end - start))
   



fig2=plt.figure()
plt.figure(figsize = (15, 10))
plt.plot(X,Y)
plt.plot(X,Z)
plt.axis([L1,L2,-10,10])


#opérations sur les axes
axes = plt.gca()
axes.xaxis.set_ticks([-L+2*L/12*i for i in range(13)])
axes.xaxis.set_tick_params(length=5, width=3,labelsize=20)
axes.yaxis.set_tick_params(length=5, width=3,labelsize=20)
plt.xlabel(r"$\lambda$",fontsize=30)
plt.ylabel(r"$\alpha(\lambda)$",fontsize=30)
#plt.title(r"Secular function for $\omega_c=$"+str(omega_c)+r" et $n=$"+str(n),fontsize=40)
plt.grid()
LocalDestinationPath = '/home/sidd/Bureau/Recherche/Landau Bernstein Scattering/Version 5' 
os.chdir(LocalDestinationPath)
plt.savefig("Secular function")
plt.show()
"""


def dichotomie(c,d,omega_c,n):
    # on suppose que f(a)f(b)<0
    Nbit,seuil=1000, 1e-10
    #N,seuil=100, 1e-8 #pour calculer xD^*
    sol=(c+d)/2.
    trouve=abs(fonc_sec(sol,omega_c,n))< seuil
    Nb=0
    for k in range(Nbit):
        Nb=Nb+1
        if Nb % 10==0:
            print(Nb)
            print(abs(fonc_sec(sol,omega_c,n)))
        if trouve:
            print(Nb)
            break
            
        elif fonc_sec(c,omega_c,n)*fonc_sec(sol,omega_c,n) >0:
            c=sol
        else:        
            d=sol
        sol=(c+d)/2.
        trouve=abs(fonc_sec(sol,omega_c,n))< seuil
    if trouve:
        return(sol)  
    else:
        return("Echec !")



"""
m=-1
if m>0:
 c=m+0.1
 d=m+0.9
if m<0:
 c=m-0.9
 d=m-0.1
"""
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
#définition du vecteur propre général pour le système linéaire
def f_mode_propre_nm(V1,V2,X,omega_c,n,lambda_m):
    def u(v1,v2):
        r2=v1**2+v2**2
        r=np.sqrt(r2)
        tau_p=np.exp(-r2/4)
        u_n=np.exp(-n*1j*v2/omega_c)
        
        #définition de la somme
        epsilon=10**(-15)
        somme=0
        m=1
        p=-1
        Tp=m*omega_c/(m*omega_c+lambda_m)*scipy.special.jv(m,n*r/omega_c)*np.exp(1j*m*np.arctan2(v2,v1))
        Tn=p*omega_c/(p*omega_c+lambda_m)*scipy.special.jv(p,n*r/omega_c)*np.exp(1j*p*np.arctan2(v2,v1))
        while abs(Tp) and abs(Tn)>epsilon:
            Tp=m*omega_c/(m*omega_c+lambda_m)*scipy.special.jv(m,n*r/omega_c)*np.exp(1j*m*np.arctan2(v2,v1))
            Tn=p*omega_c/(p*omega_c+lambda_m)*scipy.special.jv(p,n*r/omega_c)*np.exp(1j*p*np.arctan2(v2,v1))
            somme+=Tp+Tn
            m=m+1
            p=p-1
        
        return tau_p*u_n*somme
    
    f2V=np.vectorize(u)
    ff2=f2V(np.transpose([V1]),V2)
    
    
    ffx=[np.exp(n*1j*x)*ff2 for x in X]
    return ffx

#définition du vecteur propre général qui perturbe la CI pour le système non-lin
def f_mode_propre_nm_nonlin(V1,V2,X,omega_c,n,lambda_m,epsilon):
    def u(v1,v2):
        r2=v1**2+v2**2
        r=np.sqrt(r2)
        tau_p=np.exp(-r2/4)
        sqrt_f0=np.exp(-r2/4)
        u_n=np.exp(-n*1j*v2/omega_c)
        
        #définition de la somme
        epsilon=10**(-15)
        somme=0
        m=1
        p=-1
        Tp=m*omega_c/(m*omega_c+lambda_m)*scipy.special.jv(m,n*r/omega_c)*np.exp(1j*m*np.arctan2(v2,v1))
        Tn=p*omega_c/(p*omega_c+lambda_m)*scipy.special.jv(p,n*r/omega_c)*np.exp(1j*p*np.arctan2(v2,v1))
        while abs(Tp) and abs(Tn)>epsilon:
            Tp=m*omega_c/(m*omega_c+lambda_m)*scipy.special.jv(m,n*r/omega_c)*np.exp(1j*m*np.arctan2(v2,v1))
            Tn=p*omega_c/(p*omega_c+lambda_m)*scipy.special.jv(p,n*r/omega_c)*np.exp(1j*p*np.arctan2(v2,v1))
            somme+=Tp+Tn
            m=m+1
            p=p-1
        
        return sqrt_f0*tau_p*u_n*somme
    
    f2V=np.vectorize(u)
    ff2=f2V(np.transpose([V1]),V2)
    """
    def f_0(v1,v2):
        r2=v1**2+v2**2
        return np.exp(-r2/2)
    
    f0V=np.vectorize(f_0)
    ff0=f0V(np.transpose([V1]),V2)
    """
    ffx=[epsilon*np.exp(n*1j*x)*ff2 for x in X]
    return ffx
"""
#définition du vecteur propre général qui perturbe la CI pour le système non-lin IMAGINAIRE
def f_mode_propre_nm_nonlin_imag(V1,V2,X,omega_c,n,lambda_m,epsilon):
    def u(v1,v2):
        r2=v1**2+v2**2
        r=np.sqrt(r2)
        tau_p=np.exp(-r2/4)
        sqrt_f0=np.exp(-r2/4)
        u_n=np.exp(-n*1j*v2/omega_c)
        
        #définition de la somme
        epsilon=10**(-15)
        somme=0
        m=1
        p=-1
        Tp=m*omega_c/(m*omega_c+lambda_m)*scipy.special.jv(m,n*r/omega_c)*np.exp(1j*m*np.arctan2(v2,v1))
        Tn=p*omega_c/(p*omega_c+lambda_m)*scipy.special.jv(p,n*r/omega_c)*np.exp(1j*p*np.arctan2(v2,v1))
        while abs(Tp) and abs(Tn)>epsilon:
            Tp=m*omega_c/(m*omega_c+lambda_m)*scipy.special.jv(m,n*r/omega_c)*np.exp(1j*m*np.arctan2(v2,v1))
            Tn=p*omega_c/(p*omega_c+lambda_m)*scipy.special.jv(p,n*r/omega_c)*np.exp(1j*p*np.arctan2(v2,v1))
            somme+=Tp+Tn
            m=m+1
            p=p-1
        
        return sqrt_f0*tau_p*u_n*somme
    
    f2V=np.vectorize(u)
    ff2=f2V(np.transpose([V1]),V2)
    
    def f_0(v1,v2):
        r2=v1**2+v2**2
        return np.exp(-r2/2)
    
    f0V=np.vectorize(f_0)
    ff0=f0V(np.transpose([V1]),V2)
    
    ffx=[ff0+epsilon*np.imag(np.exp(n*1j*x)*ff2) for x in X]
    return ffx
"""
"""
start=time.time()
vec_propre=f_mode_propre_nm(V1,V2,X,omega_c,n,lambda_m)
#print(vec_propre)
end=time.time()

print("Elapsed time for loading init = ",(end-start))


fxv3=np.zeros((M1,M2))
fxv4=np.zeros((M1,M2))
fxv5=np.zeros((M1,M2))
for i in range(M1):
    for j in range(M2):
        fxv3[i,j]=np.real(vec_propre[n][i,j])
        fxv4[i,j]=np.imag(vec_propre[n][i,j])
        fxv5[i,j]=abs(vec_propre[n][i,j])
         
fig3=plt.figure()
#te=t*dt
plt.figure(figsize = (15, 10))
#plt.axis([0,16,-10, 10])
plt.pcolormesh(V2, V1, fxv3,cmap='RdBu')
plt.title("densité f REELLE dans le plan V1-V2 pour x="+str(0))
plt.colorbar()
plt.ylabel('V2')
plt.xlabel('V1')  
plt.legend()
plt.show()

fig4=plt.figure()
#te=t*dt
plt.figure(figsize = (15, 10))
#plt.axis([0,16,-10, 10])
plt.pcolormesh(V2, V1, fxv4,cmap='RdBu')
plt.title("densité f IMAGINAIRE dans le plan V1-V2 pour x="+str(0))
plt.colorbar()
plt.ylabel('V2')
plt.xlabel('V1')  
plt.legend()
plt.show()

fig5=plt.figure()
#te=t*dt
plt.figure(figsize = (15, 10))
#plt.axis([0,16,-10, 10])
plt.pcolormesh(V2, V1, fxv5,cmap='RdBu')
plt.title("densité f MODULE dans le plan V1-V2 pour x="+str(0))
plt.colorbar()
plt.ylabel('V2')
plt.xlabel('V1')  
plt.legend()
plt.show()
"""

"""
#verification mode propre en claculant le champ éléctrique initial grace à Poisson

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


e=Poisson_trap(vec_propre)
er=np.real(e)
ei=np.imag(e)

figE1=plt.figure()
#te=t*dt
plt.figure(figsize = (15, 10))
#plt.axis([0,16,-10, 10])
plt.plot(X, er,label="calcul")
plt.plot(X,np.sin(X),label="théorique")
plt.title("champ électrique REELLE")
plt.ylabel('E réel')
plt.xlabel('X')
axes = plt.gca()
axes.xaxis.set_tick_params(length=5, width=3,labelsize=20)
axes.yaxis.set_tick_params(length=5, width=3,labelsize=20)   
plt.legend()
plt.grid()
plt.show()

figE2=plt.figure()
#te=t*dt
plt.figure(figsize = (15, 10))
#plt.axis([0,16,-10, 10])
plt.plot(X, ei,label="calcul")
plt.plot(X,-np.cos(X),label="théorique")
plt.title("champ électrique IMAGINAIRE")
plt.ylabel('E imaginaire')
plt.xlabel('X')
axes = plt.gca()
axes.xaxis.set_tick_params(length=5, width=3,labelsize=20)
axes.yaxis.set_tick_params(length=5, width=3,labelsize=20)   
plt.legend()
plt.grid()
plt.show()
"""
"""
def norme2_nm(lam,omega_c,n):
    epsilon=10**(-15)
    somme=0
    m=1
    p=-1
    Tm=(m*omega_c)**2/(m*omega_c+lam)**2*a_sec(m,omega_c,n)[0]
    Tp=(p*omega_c)**2/(p*omega_c+lam)**2*a_sec(p,omega_c,n)[0]
    while abs(Tm) and abs(Tp)>epsilon:
        Tm=(m*omega_c)**2/(m*omega_c+lam)**2*a_sec(m,omega_c,n)[0]
        Tp=(p*omega_c)**2/(p*omega_c+lam)**2*a_sec(p,omega_c,n)[0]
        somme+=Tm+Tp
        m=m+1
        p=p-1
    #print("(m,p)=",(m,p))
    return 2*np.pi*somme+n**2

masse_quadra=0
for k in range(N-1):
    for i in range(M1-1):
        for j in range(M2-1):
            masse_quadra+=abs(vec_propre[k][i,j])**2
masse_quadra=masse_quadra*dx*dv1*dv2/L
         
masse_quadra_E=0
for k in range(N-1):
    masse_quadra_E+=abs(e[k])**2
masse_quadra_E=masse_quadra_E*dx/L
print("   énergie quadratique= ",masse_quadra+masse_quadra_E)
print("   énergie quadratique théorique= ",norme2_nm(lambda_m,omega_c,n))



def Secante(omega_c,n,Nbit,seuil,x0,x1):  
    x2=x1-fonc_sec(x1,omega_c,n)*(x1-x0)/(fonc_sec(x1,omega_c,n)-fonc_sec(x0,omega_c,n))   
    trouve=abs(fonc_sec(x2,omega_c,n))< seuil 
    Nb=0
    print(fonc_sec(x2,omega_c,n)-fonc_sec(x1,omega_c,n))      
    for k in range(Nbit):
        Nb+=1
        if trouve:
            
            break
        print(Nb)
        temp=x2
        y2=fonc_sec(x2,omega_c,n)
        z1=fonc_sec(x1,omega_c,n)
        x2=x2-y2*(x2-x1)/(y2-z1)
        x1=temp
        trouve=abs(fonc_sec(x2,omega_c,n))< seuil
    if trouve:
        return(x0)  
    else:
        return("Echec !")

start=time.time()
Nbit=100
seuil=1e-5
m=5
x0=m+0.5
x1=m+0.4
print("secan",Secante(omega_c,n,Nbit,seuil,x0,x1))          
end=time.time()
print("Elapsed time for dichotomy = ",(end-start))
"""