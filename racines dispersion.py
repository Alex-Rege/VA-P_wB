#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import scipy
import math

from scipy.special import erfi
from scipy.optimize import fsolve
"""
"""
k=0.5

def D(w):
    a= 1/(k**2)
    b=(np.pi)**(0.5)*w/(2**0.5*k)
    c=np.exp(-w**2/(2*k**2))
    d=1-scipy.special.erf(w/(2**0.5*k))
    return 1+a*(1+b*c*d)

print(D(1))
print(scipy.optimize.fsolve(D,1))

N=11
a=-1
b=4
x1=b-a
X2=np.linspace(a,b,N)
