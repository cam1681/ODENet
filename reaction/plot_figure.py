plt.figure()
plt.subplot(2,2,1)
plt.plot(true_yorig.numpy()[0:args.train_size,0,0]-true_y.numpy()[0:args.train_size,0,0],'*')
plt.ylabel('noise')
plt.subplot(2,2,2)
plt.plot(true_yorig.numpy()[0:args.train_size,0,1]-true_y.numpy()[0:args.train_size,0,1],'*')
plt.subplot(2,2,3)
plt.plot(func.err.data.numpy()[0:args.train_size,0,0],'*')
plt.ylabel('predicted noise')
plt.subplot(2,2,4)
plt.plot(func.err.data.numpy()[0:args.train_size,0,1],'*')
plt.savefig('err.jpg')


