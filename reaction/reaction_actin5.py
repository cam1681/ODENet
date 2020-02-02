
## TODO consider set data size in precompute
import os
import argparse
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib
import matplotlib.pyplot as plt
import json

parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='dopri5')
#parser.add_argument('--data_size', type=int, default=3000)
parser.add_argument('--batch_time', type=int, default=3) # case1 40
parser.add_argument('--rand_coef', type=float, default=0.0)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--batch_sample', type=int, default=10)
parser.add_argument('--niters', type=int, default=10000)
parser.add_argument('--vc_init_flag', type=int, default=0)
parser.add_argument('--variable_nums', type=int, default=2)
parser.add_argument('--ode_nums', type=int, default=2)
parser.add_argument('--CASE', type=int, default=1)
parser.add_argument('--basis_order', type=int, default=2)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--test_freq', type=int, default=10)
parser.add_argument('--viz', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
args = parser.parse_args()
vail_init_value = None

indexv = np.loadtxt('./data/indexv5.txt')
indexv = [int(_index) for _index in indexv]
vdata = np.loadtxt('./data/vdata5.txt')
args.batch_size = len(indexv)

if args.CASE >= 3:
    args.variable_nums = 3
else:
    args.variable_nums = 2

args.variable_nums = 2

from torchdiffeq import odeint

def vodeint(func, batch_y0, batch_t, ode_nums):
    pred_y = torch.stack([odeint(func, batch_y0[i], batch_t[i]) for i in range(args.batch_size*args.batch_sample)], dim=1)
    return pred_y

def reodeint(func, true_y0, vt):
    result_y = torch.cat([odeint(func, true_y0[i], vt[i], method='rk4') for i in range(args.batch_size)])
    return result_y

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
#equationdata = np.loadtxt('EquationData_{}.txt'.format(args.CASE))
equationdata = vdata
true_y = torch.from_numpy(equationdata[:,1:].reshape(-1,1,args.variable_nums)).type(torch.FloatTensor)
true_yorig = true_y
t = torch.from_numpy(equationdata[:,0]).type(torch.FloatTensor)
# Add random white noise
#true_y = torch.max(abs(true_y),0)[0]*args.rand_coef*torch.Tensor(np.random.normal(0,1,list(true_y.size()))) + true_y
true_y = torch.mul((1+args.rand_coef*torch.Tensor(np.random.normal(0,1,list(true_y.size()))) ),true_y)
#print(true_y.size())
#plt.plot(true_y.numpy()[:,0,0],true_y.numpy()[:,0,1])
#plt.plot(true_y.numpy()[:,0,0])
#plt.plot(true_yorig.numpy()[:,0,0]-true_y.numpy()[:,0,0])
#plt.plot(true_yorig.numpy()[:,0,0])
#plt.show()
#true_y = args.rand_coef*torch.Tensor(np.random.normal(0,1,list(true_y.size()))) + true_y
# Set data_size and train_size
args.data_size = t.size()[0]
args.train_size = int(args.data_size)
train_y = true_y[:args.train_size]
#test_y = true_y[args.train_size:]

vt = [t[:indexv[i]+1] if i==0 else t[indexv[i-1]+1:indexv[i]+1] for i in range(args.batch_size)]
indexint = indexv.copy()
indexint.pop()
indexint = [a+1 for a in indexint]
indexint.insert(0,0)
true_y0 = true_y[indexint]

def get_batch(train_size, batch_time, train_y, t, err_y, ode_nums, batch_sample):
    #s = torch.from_numpy(np.random.choice(np.arange(train_size - batch_time, dtype=np.int64), args.batch_size, replace=False))
    s = torch.LongTensor([np.random.choice(range(0,indexv[i]-batch_time+1),batch_sample) if i==0 else np.random.choice(range(indexv[i-1]+1,indexv[i]-batch_time+1),batch_sample) for i in range(args.batch_size)])
    s = s.reshape(-1)
    batch_y0 = train_y[s] - err_y.data[s]
    batch_t = [t[s[i]:s[i]+batch_time] for i in range(s.shape[0])]
    batch_y = torch.stack([true_y[s + i] for i in range(batch_time)], dim=0) - \
            torch.stack([err_y.data[s + i] for i in range(batch_time)], dim=0)
    return batch_y0, batch_t, batch_y


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


if args.viz:
    makedirs('png{}'.format(args.CASE))
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(12, 4), facecolor='white')
    if args.variable_nums == 3:
        ax_traj = fig.add_subplot(321, frameon=False)
        ay_traj = fig.add_subplot(322, frameon=False)
        az_traj = fig.add_subplot(323, frameon=False)
        axy_phase = fig.add_subplot(324, frameon=False)
        axz_phase = fig.add_subplot(325, frameon=False)
        ayz_phase = fig.add_subplot(326, frameon=False)
    else:
        ax_traj = fig.add_subplot(121, frameon=False)
        ay_traj = fig.add_subplot(122, frameon=False)
        #axy_phase = fig.add_subplot(122, frameon=False)



def visualize(vt, true_y, pred_y, odefunc, itr, loss):

    if args.viz:

        ax_traj.cla()
        ax_traj.set_title('Trajectories')
        ax_traj.set_xlabel('t')
        ax_traj.set_ylabel('x')
        for i in range(len(indexv)):
            if i==0:
                ax_traj.plot(vt[i].numpy(), pred_y.numpy()[:indexv[i]+1, 0, 0], 'r--', vt[i].numpy(), true_y.numpy()[:indexv[i]+1, 0, 0], 'k-')
            else:
                ax_traj.plot(vt[i].numpy(), pred_y.numpy()[indexv[i-1]+1:indexv[i]+1, 0, 0], 'r--', vt[i].numpy(), true_y.numpy()[indexv[i-1]+1:indexv[i]+1, 0, 0], 'k-')
        ax_traj.legend()

        ay_traj.cla()
        ay_traj.set_title('Trajectories')
        ay_traj.set_xlabel('t')
        ay_traj.set_ylabel('y')
        for i in range(len(indexv)):
            if i==0:
                ay_traj.plot(vt[i].numpy(), pred_y.numpy()[:indexv[i]+1, 0, 1], 'r--', vt[i].numpy(), true_y.numpy()[:indexv[i]+1, 0, 1], 'k-')
            else:
                ay_traj.plot(vt[i].numpy(), pred_y.numpy()[indexv[i-1]+1:indexv[i]+1, 0, 1], 'r--', vt[i].numpy(), true_y.numpy()[indexv[i-1]+1:indexv[i]+1, 0, 1], 'k-')
        ay_traj.legend()

        if args.variable_nums == 3:
            ay_traj.cla()
            ay_traj.set_title('Trajectories')
            ay_traj.set_xlabel('t')
            ay_traj.set_ylabel('y')
            ay_traj.plot(time.numpy(), pred_y.numpy()[:, 0, 1], 'r--', time.numpy(), true_y.numpy()[:, 0, 1], 'k-')
            ay_traj.set_xlim(time.min(), time.max())
            # ax_traj.set_ylim(-2, 2)
            ay_traj.legend()
            az_traj.cla()
            az_traj.set_title('Trajectories')
            az_traj.set_xlabel('t')
            az_traj.set_ylabel('z')
            az_traj.plot(time.numpy(), pred_y.numpy()[:, 0, 2], 'r--', time.numpy(), true_y.numpy()[:, 0, 2], 'k-')
            az_traj.set_xlim(time.min(), time.max())
            # ax_traj.set_ylim(-2, 2)
            az_traj.legend()


            axz_phase.cla()
            axz_phase.set_title('Phase Portrait')
            axz_phase.set_xlabel('x')
            axz_phase.set_ylabel('z')
            axz_phase.plot(true_y.numpy()[:, 0, 0], true_y.numpy()[:, 0, 2], 'r--')
            axz_phase.plot(pred_y.numpy()[:, 0, 0], pred_y.numpy()[:, 0, 2], 'k-')

            ayz_phase.cla()
            ayz_phase.set_title('Phase Portrait')
            ayz_phase.set_xlabel('y')
            ayz_phase.set_ylabel('z')
            ayz_phase.plot(true_y.numpy()[:, 0, 1], true_y.numpy()[:, 0, 2], 'r--')
            ayz_phase.plot(pred_y.numpy()[:, 0, 1], pred_y.numpy()[:, 0, 2], 'k-')

        fig.tight_layout()
        plt.savefig('png{}/{}.jpg'.format(args.CASE,itr))
        #plt.draw()
        #plt.pause(0.001)


class OdeNet(nn.Module):
    def __init__(self, variable_nums, ode_nums, basis_order, vc_init_flag, train_size):
        super(OdeNet, self).__init__()
        self._total_basis = int(np.math.factorial(variable_nums+basis_order)\
                /(np.math.factorial(variable_nums)*np.math.factorial(basis_order)))
        self.vc = nn.Parameter(torch.Tensor(self._total_basis,ode_nums))
        self.variable_nums = variable_nums
        self.ode_nums = ode_nums
        self.basis_order = basis_order

        self.err = nn.Parameter(torch.Tensor(train_size, 1, ode_nums))
        #self.err.data = true_y*0.01
        self.err.data.uniform_(-0.1, 0.1)
        self._vc_init_flag = vc_init_flag
        self.vc.data.uniform_(-0.1, 0.1)
        self.vc.data[0,:] = 0
        
        #self.vc.data = torch.tensor(
        #[[0.0, -0.0], [-0.07513687759637833, 0.07525459676980972], [-0.5160207152366638, 0.5171677470207214], [0.0, -0.0], [0.09321971237659454, -0.09383248537778854], [0.08520904928445816, -0.08398796617984772]])
        vail_init_value = self.vc.data.numpy()

    def forward(self, t, x_input):


        def _compute_theta3d():
            _basis_count = 0
            _Theta = torch.zeros(x_input.size(0),1,self._total_basis)
            _Theta[:,0,0] = 1
            _basis_count += 1
            for ii in range(0,self.variable_nums):
                _Theta[:,0,_basis_count] = x_input[:,0,ii]
                _basis_count += 1

            if self.basis_order >= 2:
                for ii in range(0,self.variable_nums):
                    for jj in range(ii,self.variable_nums):
                        _Theta[:,0,_basis_count] = torch.mul(x_input[:,0,ii],x_input[:,0,jj])
                        _basis_count += 1

            if self.basis_order >= 3:
                for ii in range(0,self.variable_nums):
                    for jj in range(ii,self.variable_nums):
                        for kk in range(jj,self.variable_nums):
                            _Theta[:,0,_basis_count] = torch.mul(torch.mul(x_input[:,0,ii], \
                                x_input[:,0,jj]),x_input[:,0,kk])
                            _basis_count += 1

            if self.basis_order >= 4:
                for ii in range(0,self.variable_nums):
                    for jj in range(ii,self.variable_nums):
                        for kk in range(jj,self.variable_nums):
                            for ll in range(kk,self.variable_nums):
                                _Theta[:,0,_basis_count] = torch.mul(torch.mul(torch.mul(x_input[:,0,ii],\
                                    x_input[:,0,jj]),x_input[:,0,kk]),x_input[:,0,ll])
                                _basis_count += 1

            if self.basis_order >= 5:
                for ii in range(0,self.variable_nums):
                    for jj in range(ii,self.variable_nums):
                        for kk in range(jj,self.variable_nums):
                            for ll in range(kk,self.variable_nums):
                                for mm in range(ll,self.variable_nums):
                                    _Theta[:,0,_basis_count] = torch.mul(torch.mul(torch.mul(torch.mul(\
                                        x_input[:,0,ii],x_input[:,0,jj]),x_input[:,0,kk]),\
                                            x_input[:,0,ll]),x_input[:,0,mm])
                                    _basis_count += 1
            assert _basis_count == self._total_basis
            return _Theta

        def _compute_theta2d():
            _basis_count = 0
            _Theta = torch.zeros(x_input.size(0),self._total_basis)
            _Theta[:,0] = 1
            _basis_count += 1
            for ii in range(0,self.variable_nums):
                _Theta[:,_basis_count] = x_input[:,ii]
                _basis_count += 1

            if self.basis_order >= 2:
                for ii in range(0,self.variable_nums):
                    for jj in range(ii,self.variable_nums):
                        _Theta[:,_basis_count] = x_input[:,ii]*x_input[:,jj]
                        _basis_count += 1

            if self.basis_order >= 3:
                for ii in range(0,self.variable_nums):
                    for jj in range(ii,self.variable_nums):
                        for kk in range(jj,self.variable_nums):
                            _Theta[:,_basis_count] = x_input[:,ii]*x_input[:,jj]*x_input[:,kk]
                            _basis_count += 1

            if self.basis_order >= 4:
                for ii in range(0,self.variable_nums):
                    for jj in range(ii,self.variable_nums):
                        for kk in range(jj,self.variable_nums):
                            for ll in range(kk,self.variable_nums):
                                _Theta[:,_basis_count] = x_input[:,ii]*x_input[:,jj]*x_input[:,kk]*x_input[:,ll]
                                _basis_count += 1

            if self.basis_order >= 5:
                for ii in range(0,self.variable_nums):
                    for jj in range(ii,self.variable_nums):
                        for kk in range(jj,self.variable_nums):
                            for ll in range(kk,self.variable_nums):
                                for mm in range(ll,self.variable_nums):
                                    _Theta[:,_basis_count] = x_input[:,ii]*x_input[:,jj]*\
                                    x_input[:,kk]*x_input[:,ll]*x_input[:,mm]
                                    _basis_count += 1
            assert _basis_count == self._total_basis
            return _Theta

        if x_input.dim() == 2:
            output = torch.mm(_compute_theta2d(),self.vc)
        else:
            output = torch.matmul(_compute_theta3d(),self.vc)

        return output

def saveStateDict(input,itr,CASE):
    """A function used to save model.state_dict() OrderedDict as json file with 'matrxi%s.json'%itr
    file name.
    Example: saveStateDict(func.state_dict(), 2, 2) creats matrix2.json in matrix2 directory.
    """
    input_dict = dict()
    for k,v in input.items():
        input_dict[k] = v.numpy().tolist()
    if not os.path.exists('matrix{}'.format(CASE)):
        os.makedirs('matrix{}'.format(CASE))
    with open('matrix{}/matrix{}.json'.format(CASE,itr), 'w') as fp:
        json.dump(input_dict, fp)

class ShrinkParameter(object):

    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, module):
        if hasattr(module, 'vc'):
            w = module.vc.data
            #print(module.vc)
            _indexzero = abs(w)/abs(w).max() < self.threshold
            module.vc.data[_indexzero] = 0


class Shrink(object):

    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, module, threshold):
        self.threshold = threshold
        if hasattr(module, 'vc'):
            w = module.vc.data
            _indexzero = abs(w) < self.threshold
            module.vc.data[_indexzero] = 0
            module.vc.grad[_indexzero] = 0


def adjust_learning_rate(optimizer, itr):
    """Sets the learning rate to the initial LR decayed by 0.5 every 1000 epochs"""
    lr = args.lr * (0.9 ** (itr // 1000))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':

    ii = 0

    func = OdeNet(args.variable_nums, args.ode_nums,args.basis_order,args.vc_init_flag, args.train_size)
    func.float()
    # 0.005 0.001
    shrinkparameter = ShrinkParameter(0.00)
    shrink = Shrink(0.0000)

    params = list(func.parameters())
    optimizer = optim.Adam(\
            [\
        {"params": func.vc, "lr": args.lr},\
        {"params": func.err, "lr": args.lr},\
        ],lr=5e-4, betas=(0.6, 0.999), eps=1e-08, weight_decay=0, amsgrad=False) #CASE 1 betas = 0.6


    loss_train_list = list([])
    loss_batch_list = list([])

    former_loss_train = 100
    for itr in range(1, args.niters + 1):
        optimizer.zero_grad()
        func.vc.data[0,:]=0
        func.vc.data[:,1] = -func.vc.data[:,0]
        batch_y0, batch_t, batch_y = get_batch(args.train_size, args.batch_time, train_y, t, func.err, args.ode_nums, args.batch_sample)
        pred_y = vodeint(func, batch_y0, batch_t, args.ode_nums)
        loss = torch.mean(torch.abs(pred_y - batch_y))
        l1_regularization = torch.tensor(0).type(torch.FloatTensor) 
        l1_regularization = l1_regularization + torch.norm(func.vc.data,1) + torch.norm(func.err.data,1)
        #factor =  0.01 if 0.005*1.2**(itr/1000) > 0.01 else 0.005*1.3**(itr/1000)
        factor =  0.0001
        loss += factor * l1_regularization

        loss_batch_list.append([itr,loss.data.item()])
        #if itr == 10000:
            #np.savetxt('loss_batch{}.txt'.format(args.CASE), loss_batch_list)

        print(itr)
        print(func.vc.data)
        print(loss)

        loss.backward()
        #if itr<5000:
        #func.apply(shrink)
        #shrink(func,0.0001 if 0.00001*2*itr/1000>0.0001 else 0.00001*2*itr/1000)
        func.vc.grad[:,1] = 0

        adjust_learning_rate(optimizer, itr)
        optimizer.step()

        if itr % args.test_freq == 0:
            with torch.no_grad():
                #pred_test = vodeint(func, test_y[0], t[len(train_y):])
                #loss_test = torch.mean(torch.abs(pred_test - test_y))
                #print('Iter {:04d} | test Loss {:.6f}'.format(itr, loss_test.item()))
                pred_train = reodeint(func, true_y0, vt)
                loss_train = torch.mean(torch.abs(pred_train - train_y))
                if torch.isnan(loss_train):
                    continue
                if loss_train>former_loss_train:
                    former_loss_train = loss_train.clone()
                    continue
                former_loss_train = loss_train.clone()
                #print('Iter {:04d} | train Loss {:.6f}'.format(itr, loss_train.item()))
                # update args.test_freq
                if loss_train > 1:
                    args.test_freq = 10
                    args.lr = 0.01
                if loss_train < 1:
                    args.test_freq = 1
                    args.lr = 0.001
                if loss_train < 0.5:
                    args.test_freq = 10
                    args.lr = 0.0001
                if loss_train < 0.2:
                    args.lr = 0.0001
                    args.test_freq = 1
                # save (itr,loss) pair
                loss_train_list.append([itr,loss_train.data.item()])
                np.savetxt('loss_train{}.txt'.format(args.CASE), loss_train_list)

                #visualize(t[len(train_y):], test_y, pred_test, func, itr, loss_test.item())
                visualize(vt, train_y, pred_train, func, itr, loss_train.item())

                #pred_all_y = odeint(func, true_y0, t)
                #visualize(t, true_y, pred_all_y, func, ii)
                #visualize(t, true_y, pred_all_y, func, ii)

                ii += 1

                #print('Learning iteration is: ', w)
                #Write the parameters to json file
                print(func.vc.data)
                print('loss_train: {}'.format(loss_train))
                #print(func.err.data)
                saveStateDict(func.state_dict(), itr, args.CASE)
        end = time.time()
    #print(vail_init_value)

# plot and save loss_itr figure

loss_data=np.loadtxt('loss_train{}.txt'.format(args.CASE))
plt.figure()
plt.plot(loss_data[:,0],loss_data[:,1])
plt.xlabel('itr')
plt.ylabel('total train set loss')
plt.savefig('loss_train{}.jpg'.format(args.CASE))

# find corresponding coeficient matrix
ind = np.argmin(loss_data,axis=0)
print('The minist loss:',loss_data[ind[1]])
