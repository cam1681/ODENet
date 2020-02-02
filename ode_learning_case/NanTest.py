""" Test the initial condition and learn why it is corrupted for some initials. Created by pi 2019/06/08"""


import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import os

CASE_CHOOSE = 2;

def rk4(r, t, h):
    """ Runge-Kutta 4 method """
    k1 = h*f(r, t)
    k2 = h*f(r+0.5*k1, t+0.5*h)
    k3 = h*f(r+0.5*k2, t+0.5*h)
    k4 = h*f(r+k3, t+h)
    return (k1 + 2*k2 + 2*k3 + k4)/6

def f(r, t):
    x, y = r[0], r[1]
    if CASE_CHOOSE == 0:
        fxd = x - 0.00029*x*x - 0.003*x*y
        fyd = 0.4*y + 0.00044*x*y - 0.0008*y*y
    elif CASE_CHOOSE == 1:
        fxd = 2*x - 0.1*x*x - 1.1*x*y
        fyd = -y - 0.1*y*y + 0.9*x*y
    elif CASE_CHOOSE == 2:
        fxd = 0.0631*x-0.0527*y+0.0916*x*x-0.0117*x*y - 0.0143*y*y
        fyd = -0.0559*x-0.0278*y-0.0971*x*x+0.0902*x*y - 0.0704*y*y
    else:
        raise ValueError('CASE_CHOOSE must be in the set {0,1,2}')
    return np.array([fxd, fyd], float)

xpoints, ypoints  = [], []
if CASE_CHOOSE == 0:
    # Time step
    h=0.001
    # t series
    tpoints = np.arange(0, 40, h)
    # Initial value
    r = np.array([100, 50], float)
    jump = 10;
elif CASE_CHOOSE == 1:
    h=0.001
    tpoints = np.arange(0, 30, h)
    r = np.array([1, 0.5], float)
    jump = 10;
elif CASE_CHOOSE == 2:
    h=0.001
    tpoints = np.arange(0, 40, h)
    r = np.array([100, 20], float)
    jump = 10;
else:
    raise ValueError('CASE_CHOOSE must be in the set {0,1,2}')

for t in tpoints:
    xpoints.append(r[0])
    ypoints.append(r[1])
    r += rk4(r, t, h)


fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(tpoints, xpoints)
ax1.plot(tpoints, ypoints)
ax1.set_title('time evol')
ax2.plot(xpoints, ypoints)
ax2.set_title('phase')
plt.show()
