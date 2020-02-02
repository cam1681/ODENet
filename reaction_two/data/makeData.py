import numpy as np
import matplotlib.pylab as plt
from itertools import chain, product
N = 3;

rawdata = np.loadtxt('actin{}.txt'.format(N),unpack=False)
rawdata[:,0] = rawdata[:,0]/3600
indices,  = np.where((np.diff(rawdata[:,0])==0)*1 == 1)
x = np.delete(rawdata[:,0],indices)
y = np.delete(rawdata[:,1],indices)





maxindex = x.shape[0]

diffx = np.sign(np.diff(x))
for i in range(len(diffx)):
    if diffx[i] == 0:
        diffx[i] = diffx[i-1]
#diffx = [_df if _df != 0 else 1  for _df in diffx]
shift = (np.diff(diffx) !=0 )*1
index, = np.where(shift == 1)
index[1::2] += 1


xs = index.shape[0] + 1
xdata = [[]]*xs
ydata = [[]]*xs

for i in range(xs):
    if i == 0:
        xdata[i] = x[:index[i]]
        ydata[i] = y[:index[i]]
    elif i == xs-1:
        xdata[i] = x[index[i-1]+1:]
        ydata[i] = y[index[i-1]+1:]
    else:
        xdata[i] = x[index[i-1]+1:index[i]]
        ydata[i] = y[index[i-1]+1:index[i]]

    if i%2 != 0:
        xdata[i] = xdata[i][:0:-1]
        ydata[i] = ydata[i][:0:-1]

vx = list(chain.from_iterable(xdata))
vy = list(chain.from_iterable(ydata))

vdata = np.array([list(_a) for _a in zip(vx,vy)])


dvx = np.sign(np.diff(vx))
indexv, = np.where(dvx==-1)
indexv = np.append(indexv,len(vx)-1)

if N == 3:
    vc = [7.4, 9.6, 12.4, 14.2, 16.2, 18.4, 20.5]
    vc.reverse()
else:
    vc = [6.7, 8.5, 11.5, 14.9, 17.3, 20.3, 22.9]
    vc.reverse()

vcd = list(list(np.repeat(vc[i],indexv[i]+1)) if i==0 else list(np.repeat(vc[i],indexv[i]-indexv[i-1])) for i in range(len(indexv)))
vcd = np.array(list(chain.from_iterable(vcd)))
vdata = np.array([[vdata[i,0],vdata[i,1]/vcd[i],(vcd[i]-vdata[i,1])/vcd[i]] for i in range(len(vcd))])
#vdata = np.array([[vdata[i,0],vdata[i,1],vcd[i]-vdata[i,1]] for i in range(len(vcd))])


np.savetxt('vdata{}.txt'.format(N), vdata)
np.savetxt('indexv{}.txt'.format(N), indexv)

#print(xdata)
#print(ydata)
#
show = 2
for i in range(len(indexv)):
    if i == 0:
        plt.plot(vdata[:indexv[i],show])
    else:
        plt.plot(vdata[indexv[i-1]+1:indexv[i]+1,show])

plt.show()







