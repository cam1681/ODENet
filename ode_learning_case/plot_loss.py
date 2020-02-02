import numpy as np
import matplotlib.pylab as plt

CASE = 3

# plot and save loss_itr figure

loss_data=np.loadtxt('loss_train{}.txt'.format(CASE))
loss_data=loss_data[0:500,:]
plt.figure()
plt.plot(loss_data[:,0],loss_data[:,1])
plt.xlabel('itr')
plt.ylabel('total train set loss')
plt.savefig('loss_train{}.jpg'.format(CASE))

# find corresponding coeficient matrix
ind = np.argmin(loss_data,axis=0)
print('The minist loss:',loss_data[ind[1]])

