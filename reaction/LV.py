""" Compute lv model's data. Created by pi 2019/05/29"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os

CASE_CHOOSE = 0;

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
        fxd = x*(1.5-y)-x*x
        fyd = -y*(1-x)
    elif CASE_CHOOSE == 1:
        fxd = 2*x - 0.1*x*x - 1.1*x*y
        fyd = -y - 0.1*y*y + 0.9*x*y
    elif CASE_CHOOSE == 2:
        fxd = x - 0.05*x*y
        fyd = 0.03*x*y - y
    elif CASE_CHOOSE >= 3 and CASE_CHOOSE <= 7:
        z = r[2]
        fxd = 10*(y-x)
        if CASE_CHOOSE == 3:
            fyd = x*(28-z)-y
        elif CASE_CHOOSE == 4:
            fyd = x*(24-z)-y
        elif CASE_CHOOSE == 5:
            fyd = x*(20-z)-y
        elif CASE_CHOOSE == 6:
            fyd = x*(10-z)-y
        elif CASE_CHOOSE == 7:
            fyd = x*(0.5-z)-y
        fzd = x*y-8/3*z
        return np.array([fxd,fyd,fzd],float)
    else:
        raise ValueError('CASE_CHOOSE must be in the set {0,1,2,3,4,5,6,7}')
    return np.array([fxd, fyd], float)

xpoints, ypoints  = [], []
zpoints = []
if CASE_CHOOSE == 0:
    # Time step
    h=0.001
    # t series
    tpoints = np.arange(0, 20, h)
    # Initial value
    r = np.array([4, 2], float)
    jump = 50;
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
elif CASE_CHOOSE >= 3 and CASE_CHOOSE <= 7:
    h=0.001
    jump = 10;
    if CASE_CHOOSE == 3:
        tpoints = np.arange(0, 40, h)
        r = np.array([0, 2, 9], float)
    elif CASE_CHOOSE == 4:
        tpoints = np.arange(0, 40, h)
        r = np.array([0, 2, 9], float)
    elif CASE_CHOOSE == 5:
        tpoints = np.arange(0, 40, h)
        r = np.array([0, 2, 9], float)
    elif CASE_CHOOSE == 6:
        tpoints = np.arange(0, 8, h)
        r = np.array([0, 2, 9], float)
    elif CASE_CHOOSE == 7:
        tpoints = np.arange(0, 5, h)
        r = np.array([0, 2, 9], float)

else:
    raise ValueError('CASE_CHOOSE must be in the set {0,1,2,3,4,5,6,7}')

for t in tpoints:
    xpoints.append(r[0])
    ypoints.append(r[1])
    if len(r) == 3:
        zpoints.append(r[2])
    r += rk4(r, t, h)

if len(r) == 2:
    equationdata = np.array(list(zip(tpoints[0:-1:jump],xpoints[0:-1:jump],ypoints[0:-1:jump])))
elif len(r) == 3:
    equationdata = np.array(list(zip(tpoints[0:-1:jump],xpoints[0:-1:jump],ypoints[0:-1:jump], zpoints[0:-1:jump])))

np.savetxt('EquationData_'+str(CASE_CHOOSE)+'.txt',equationdata)

if len(r) == 2:
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(tpoints, xpoints)
    ax1.plot(tpoints, ypoints)
    ax1.set_title('time evol')
    ax2.plot(xpoints, ypoints)
    ax2.set_title('phase')
    if not os.path.exists('pdf'):
        os.makedirs('pdf')
    fig.savefig('pdf/lv{}.pdf'.format(CASE_CHOOSE), bbox_inches='tight')
    plt.show()
elif len(r) == 3:

    fig = plt.figure(figsize=(12, 4), facecolor='white')
    ax1 = fig.add_subplot(321, frameon=False)
    ax2 = fig.add_subplot(323, frameon=False)
    ax3 = fig.add_subplot(325, frameon=False)
    ax4 = fig.add_subplot(322, frameon=False)
    ax5 = fig.add_subplot(324, frameon=False)
    ax6 = fig.add_subplot(326, frameon=False)
    ax1.plot(tpoints, xpoints)
    ax2.plot(tpoints, ypoints)
    ax3.plot(tpoints, zpoints)
    ax4.plot(xpoints, ypoints)
    ax5.plot(xpoints, zpoints)
    ax6.plot(ypoints, zpoints)
    #ax1.set_title('t-x')
    #ax2.set_title('t-y')
    #ax3.set_title('t-z')
    if not os.path.exists('pdf'):
        os.makedirs('pdf')
    fig.savefig('pdf/lv{}.pdf'.format(CASE_CHOOSE), bbox_inches='tight')
    plt.show()
