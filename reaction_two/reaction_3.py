
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
parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='adams')
#parser.add_argument('--data_size', type=int, default=3000)
parser.add_argument('--batch_time', type=int, default=4)
parser.add_argument('--rand_coef', type=float, default=0.0)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--batch_sample', type=int, default=20)
parser.add_argument('--niters', type=int, default=10000)
parser.add_argument('--vc_init_flag', type=int, default=0)
parser.add_argument('--variable_nums', type=int, default=2)
parser.add_argument('--ode_nums', type=int, default=3)
parser.add_argument('--CASE', type=int, default=3)
parser.add_argument('--basis_order', type=int, default=2)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--test_freq', type=int, default=10)
parser.add_argument('--viz', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
args = parser.parse_args()
vail_init_value = None

indexv = np.loadtxt('./data/indexv3.txt')
indexv = [int(_index) for _index in indexv]
args.batch_size = len(indexv)

equationdata = np.loadtxt('./data/actin3p.txt',delimiter=',')
TPM = np.loadtxt('./data/TPM.txt')
#equationdata[:,1] = TPM[:,1]

args.variable_nums = 3

from torchdiffeq import odeint

def vodeint(func, batch_y0, batch_t, ode_nums):
    pred_y = torch.stack([odeint(func, batch_y0[i], batch_t[i]) for i in range(args.batch_size*args.batch_sample)], dim=1)
    return pred_y

def reodeint(func, true_y0, vt):
    result_y = torch.cat([odeint(func, true_y0[i], vt[i], method='rk4') for i in range(args.batch_size)])
    return result_y

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
true_ym = torch.from_numpy(equationdata[:,1].reshape(-1,1,1)).type(torch.FloatTensor)
true_y2 = torch.from_numpy(equationdata[:,2:4].reshape(-1,1,2)).type(torch.FloatTensor)
t = torch.from_numpy(equationdata[:,0]).type(torch.FloatTensor)
# Add random white noise
# Set data_size and train_size
args.data_size = t.shape[0]
args.train_size = int(args.data_size)

vt = [t[:indexv[i]+1] if i==0 else t[indexv[i-1]+1:indexv[i]+1] for i in range(args.batch_size)]
indexint = indexv.copy()
indexint.pop()
indexint = [a+1 for a in indexint]
indexint.insert(0,0)

def update_true_y(true_y1, true_y2):
    true_y = torch.cat([true_y1, true_y2],dim=2)
    return true_y



def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


if args.viz:
    makedirs('png{}'.format(args.CASE))
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(12, 4), facecolor='white')
    if args.variable_nums > 3:
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

        if args.variable_nums > 3:
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
    def __init__(self, variable_nums, ode_nums, basis_order, vc_init_flag, train_size, true_ym):
        super(OdeNet, self).__init__()
        self._total_basis = int(np.math.factorial(variable_nums+basis_order)\
                /(np.math.factorial(variable_nums)*np.math.factorial(basis_order)))
        self.vc = nn.Parameter(torch.Tensor(self._total_basis,ode_nums))
        self.variable_nums = variable_nums
        self.ode_nums = ode_nums
        self.basis_order = basis_order

        self.true_y1 = nn.Parameter(torch.Tensor(train_size, 1, ode_nums))
        self.true_y1.data = true_ym
        self._vc_init_flag = vc_init_flag
        self.vc.data.uniform_(0, 0)
        self.vc.data[9, 0] = 0.0108
        self.vc.data[1, 1] = -2.2982
        self.vc.data[6, 1] = 1.0306
        self.vc.data[9, 1] = 2*0.0108

        #self.vc.data = torch.tensor(
        #[[0.0, -0.0], [-0.005216, 102], [-0.0118, 20], [-0.00015166,10], [-10000, 3], [-16000, 0]])
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
            self._input_size_flag = 2
            output = torch.mm(_compute_theta2d(), self.vc)
        else:
            self._input_size_flag = 3
            output = torch.matmul(_compute_theta3d(), self.vc)

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


class Shrink(object):

    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, module, threshold):
        self.threshold = threshold
        if hasattr(module, 'vc'):
            w = module.vc.data
            _indexzero = abs(w) < self.threshold
            _indexzero[0,0] = 0
            _indexzero[2,0] = 0
            _indexzero[3,0] = 0
            _indexzero[9,0] = 0
            _indexzero[2,1] = 0
            _indexzero[6,1] = 0
            module.vc.data[_indexzero] = 0
            module.vc.grad[_indexzero] = 0

class AboveBottom(object):

    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, module, threshold):
        self.threshold = threshold
        if hasattr(module, 'true_y1'):
            w = module.true_y1.data
            _indexzero = w < self.threshold
            module.true_y1.data[_indexzero] = self.threshold

def adjust_learning_rate(optimizer, itr):
    """Sets the learning rate to the initial LR decayed by 0.5 every 1000 epochs"""
    lr = args.lr * (0.9 ** (itr // 1000))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':

    ii = 0
    func = OdeNet(args.variable_nums, args.ode_nums,args.basis_order,args.vc_init_flag, args.train_size, true_ym)

    shrink = Shrink(0.01)
    above = AboveBottom(0.001)

    func.float()
    true_y = update_true_y(func.true_y1.data, true_y2)
    true_y0 = true_y[indexint]
    train_y = true_y[:args.train_size]


    params = list(func.parameters())
    optimizer = optim.Adam(\
            [\
        {"params": func.vc, "lr": 0.1*args.lr},\
        {"params": func.true_y1, "lr": args.lr},\
        ], betas=(0.6, 0.999), eps=1e-08, weight_decay=0, amsgrad=False) #CASE 1 betas = 0.6


    loss_train_list = list([])
    loss_batch_list = list([])

    former_loss_train = 10000
    for itr in range(1, args.niters + 1):
        optimizer.zero_grad()
        func.vc.data[:,2] = -func.vc.data[:,1]
        s = torch.LongTensor([np.random.choice(range(0, indexv[i]-args.batch_time+1), args.batch_sample) if i == 0 else np.random.choice(range(indexv[i-1]+1,indexv[i]-args.batch_time+1), args.batch_sample) for i in range(args.batch_size)])
        s = s.reshape(-1)
        batch_y0= train_y[s]
        batch_t = [t[s[i]:s[i]+args.batch_time] for i in range(s.shape[0])]
        batch_y = torch.stack([train_y[s + i] for i in range(args.batch_time)], dim=0)

        pred_y = vodeint(func, batch_y0, batch_t, args.ode_nums)
        loss = torch.mean(torch.abs(pred_y[:,:,:,1] - batch_y[:,:,:,1]))
        l1_regularization = torch.tensor(0).type(torch.FloatTensor)
        l1_regularization = l1_regularization + torch.norm(func.vc,1)
        factor =  0.0000
        loss += factor * l1_regularization

        loss_batch_list.append([itr,loss.data.item()])
        #if itr == 10000:
            #np.savetxt('loss_batch{}.txt'.format(args.CASE), loss_batch_list)

        #print(itr)
        #print(func.vc.data)
        #print(loss)

        loss.backward()
        #if itr<5000:
        if itr > 00:
            above(func,0.001)

        shrink(func,0.01)
        #print(func.true_y2)
        func.vc.grad[:,2] = 0
        adjust_learning_rate(optimizer, itr)
        optimizer.step()

        if itr % args.test_freq == 0:
            with torch.no_grad():
                pred_train = reodeint(func, true_y0, vt)
                loss_train = torch.mean(torch.abs(pred_train[:,:,1] - train_y[:,:,1]))
                if torch.isnan(loss_train):
                    continue
                if loss_train>former_loss_train:
                    former_loss_train = loss_train.clone()
                    continue

                #with torch.no_grad():
                func.true_y1.data = pred_train[:,0,0].reshape(-1,1,1)
                true_y = update_true_y(func.true_y1.data, true_y2)
                true_y0 = true_y[indexint]
                train_y = true_y[:args.train_size]
                #if itr > 00:
                    #above(func,0.001)

                    #true_y = update_true_y(func.true_y1.data, true_y2)
                    #true_y0 = true_y[indexint]
                    #train_y = true_y[:args.train_size]

                former_loss_train = loss_train.clone()
                #print('Iter {:04d} | train Loss {:.6f}'.format(itr, loss_train.item()))
                # update args.test_freq
                if loss_train > 0.5:
                    args.test_freq = 10
                    args.lr = 1e-3
                if loss_train < 0.5:
                    args.test_freq = 10
                    args.lr = 5e-4
                if loss_train < 0.1:
                    args.test_freq = 1
                    args.lr = 1e-4
                if loss_train < 0.05:
                    args.lr = 1e-5
                    args.test_freq = 1
                # save (itr,loss) pair
                loss_train_list.append([itr,loss_train.data.item()])
                np.savetxt('loss_train{}.txt'.format(args.CASE), loss_train_list)

                #visualize(t[len(train_y):], test_y, pred_test, func, itr, loss_test.item())
                visualize(vt, train_y.detach(), pred_train.detach(), func, itr, loss_train.item())
                tt = t.numpy()[:len(train_y)].reshape(-1,1)
                datay = np.append(tt,train_y.numpy()[:,0,:],axis=1)
                datay = np.append(datay,pred_train.numpy()[:,0,:],axis=1)
                np.savetxt('png{}/data{}.txt'.format(args.CASE,itr), datay)

                #pred_all_y = odeint(func, true_y0, t)
                #visualize(t, true_y, pred_all_y, func, ii)
                #visualize(t, true_y, pred_all_y, func, ii)

                ii += 1

                #print('Learning iteration is: ', w)
                #Write the parameters to json file
                print(func.vc.data)
                print('loss_train: {}'.format(loss_train))
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
