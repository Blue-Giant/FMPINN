"""
@author: LXA
 Date: 2021 年 11 月 11 日
 Modifying on 2022 年 9月 2 日 ~~~~ 2023 年 1月 15 日
 Final version: 2023年 2 月 1 日
"""
import os
import sys
import torch
import torch.nn as tn
import numpy as np
import matplotlib
import platform
import shutil
import time
import itertools

from Networks import DNN_base

from Problems import General_Laplace
from Problems import MS_LaplaceEqs
from Problems import MS_BoltzmannEqs

from utilizers import dataUtilizer2torch
from utilizers import DNN_tools
from utilizers import DNN_Log_Print
from utilizers import plotData
from utilizers import saveData
from utilizers import Load_data2Mat


class FMPINN(tn.Module):
    def __init__(self, input_dim=8, out_dim=1, hidden_layer=None, Model_name='DNN', name2actIn='relu',
                 name2actHidden='relu', name2actOut='linear', opt2regular_WB='L2', type2numeric='float32',
                 factor2freq=None, sFourier=1.0, repeat_highFreq=True, use_gpu=False, No2GPU=0):
        """
        initialing the class of FMPINN with given setups
        Args:
             input_dim:        the dimension of input data
             out_dim:          the dimension of output data
             hidden_layer:     the number of units for hidden layers(a tuple or a list)
             Model_name:       the name of DNN model(DNN, ScaleDNN or FourierDNN)
             name2actIn:       the name of activation function for input layer
             name2actHidden:   the name of activation function for all hidden layers
             name2actOut:      the name of activation function for all output layer
             opt2regular_WB:   the option of regularing weights and biases
             type2numeric:     the numerical type of float
             factor2freq:      the scale vector for ScaleDNN or FourierDNN
             sFourier:         the relaxation factor for FourierDNN
             repeat_highFreq:  repeat the high-frequency scale-factor or not
             use_gpu:          using cuda or not
             No2GPU:           if your computer have more than one GPU, please assign the number of GPU
        """
        super(FMPINN, self).__init__()
        self.DNN = DNN_base.Fourier_SubNets3D(
            indim=input_dim, outdim=out_dim, hidden_units=hidden_layer, name2Model=Model_name,
            actName2in=name2actIn, actName=name2actHidden, actName2out=name2actOut, type2float=type2numeric,
            repeat_Highfreq=repeat_highFreq, to_gpu=use_gpu, gpu_no=No2GPU, num2subnets=len(factor2freq))
        # opt2contri_subnets = 'sum_inv2scale'
        # opt2contri_subnets = 'five_mean_inv2scale'
        # opt2contri_subnets = 'ten_mean_inv2scale'
        # opt2contri_subnets = 'mean_inv2scale'
        # opt2contri_subnets = 'softmax_inv2scale'
        # opt2contri_subnets = 'sum_inv2subnet_order'
        # opt2contri_subnets = 'mean_inv2subnet_order'
        # opt2contri_subnets = 'softmax_inv2subnet_order'
        # opt2contri_subnets = 'random_ceof'
        # opt2contri_subnets = 'softmax_random_ceof'
        # opt2contri_subnets = 'equal_weights'

        self.input_dim = input_dim
        self.out_dim = out_dim
        self.hidden_layer = hidden_layer
        self.Model_name = Model_name
        self.name2actIn = name2actIn
        self.name2actHidden = name2actHidden
        self.name2actOut = name2actOut
        self.factor2freq = factor2freq
        self.sFourier = sFourier
        self.opt2regular_WB = opt2regular_WB

        if type2numeric == 'float32':
            self.float_type = torch.float32
        elif type2numeric == 'float64':
            self.float_type = torch.float64
        elif type2numeric == 'float16':
            self.float_type = torch.float16

        self.use_gpu = use_gpu
        if use_gpu:
            self.opt2device = 'cuda:' + str(No2GPU)
        else:
            self.opt2device = 'cpu'

        self.mat2U = torch.tensor([[1], [0], [0], [0], [0], [0], [0], [0], [0]],
                                  dtype=self.float_type, device=self.opt2device)  # 9 行 1 列
        self.mat2PX = torch.tensor([[0], [1], [0], [0], [0], [0], [0], [0], [0]],
                                   dtype=self.float_type, device=self.opt2device)  # 9 行 1 列
        self.mat2PY = torch.tensor([[0], [0], [1], [0], [0], [0], [0], [0], [0]],
                                   dtype=self.float_type, device=self.opt2device)  # 9 行 1 列
        self.mat2PZ = torch.tensor([[0], [0], [0], [1], [0], [0], [0], [0], [0]],
                                   dtype=self.float_type, device=self.opt2device)  # 9 行 1 列
        self.mat2PR = torch.tensor([[0], [0], [0], [0], [1], [0], [0], [0], [0]],
                                   dtype=self.float_type, device=self.opt2device)  # 9 行 1 列
        self.mat2PS = torch.tensor([[0], [0], [0], [0], [0], [1], [0], [0], [0]],
                                   dtype=self.float_type, device=self.opt2device)  # 9 行 1 列
        self.mat2PT = torch.tensor([[0], [0], [0], [0], [0], [0], [1], [0], [0]],
                                   dtype=self.float_type, device=self.opt2device)  # 9 行 1 列
        self.mat2PV = torch.tensor([[0], [0], [0], [0], [0], [0], [0], [1], [0]],
                                   dtype=self.float_type, device=self.opt2device)  # 9 行 1 列
        self.mat2PW = torch.tensor([[0], [0], [0], [0], [0], [0], [0], [0], [1]],
                                   dtype=self.float_type, device=self.opt2device)  # 9 行 1 列

    def loss_in2pLaplace(self, XYZRSTVW=None, fside=None, if_lambda2fside=True, aeps=None, if_lambda2aeps=True,
                        loss_type='l2_loss', scale2lncosh=0.1):
        """
        Calculating the loss of Laplace equation (*) in the interior points for given domain
        -Laplace U = f,  in Omega
        U = g            on Partial Omega
        Args:
             XYZST:           the input data of variable. ------  float, shape=[B,D]
             fside:           the force side              ------  float, shape=[B,1]
             if_lambda2fside: the force-side is a lambda function or an array  ------ Bool
             loss_type:       the type of loss function(Ritz, L2, Lncosh)      ------ string
             scale2lncosh:    if the loss is lncosh, using it                  ------- float
        return:
             UNN:             the output data
             loss_in:         the output loss in the interior points for given domain
        """
        assert (XYZRSTVW is not None)
        assert (fside is not None)

        shape2XYZRSTVW = XYZRSTVW.shape
        lenght2XYZST_shape = len(shape2XYZRSTVW)
        assert (lenght2XYZST_shape == 2)
        assert (shape2XYZRSTVW[-1] == 8)
        X = torch.reshape(XYZRSTVW[:, 0], shape=[-1, 1])
        Y = torch.reshape(XYZRSTVW[:, 1], shape=[-1, 1])
        Z = torch.reshape(XYZRSTVW[:, 2], shape=[-1, 1])
        R = torch.reshape(XYZRSTVW[:, 3], shape=[-1, 1])
        S = torch.reshape(XYZRSTVW[:, 4], shape=[-1, 1])
        T = torch.reshape(XYZRSTVW[:, 5], shape=[-1, 1])
        V = torch.reshape(XYZRSTVW[:, 6], shape=[-1, 1])
        W = torch.reshape(XYZRSTVW[:, 7], shape=[-1, 1])

        if if_lambda2fside:
            force_side = fside(X, Y, Z, R, S, T, V, W)
        else:
            force_side = fside

        UPNN = self.DNN(XYZRSTVW, scale=self.factor2freq, sFourier=self.sFourier)
        UNN = torch.matmul(UPNN, self.mat2U)
        grad2UNN = torch.autograd.grad(UNN, XYZRSTVW, grad_outputs=torch.ones_like(X), create_graph=True,
                                       retain_graph=True)
        dUNN = grad2UNN[0]
        dUNN2dx = torch.reshape(dUNN[:, 0], shape=[-1, 1])
        dUNN2dy = torch.reshape(dUNN[:, 1], shape=[-1, 1])
        dUNN2dz = torch.reshape(dUNN[:, 2], shape=[-1, 1])
        dUNN2dr = torch.reshape(dUNN[:, 3], shape=[-1, 1])
        dUNN2ds = torch.reshape(dUNN[:, 4], shape=[-1, 1])
        dUNN2dt = torch.reshape(dUNN[:, 5], shape=[-1, 1])
        dUNN2dv = torch.reshape(dUNN[:, 6], shape=[-1, 1])
        dUNN2dw = torch.reshape(dUNN[:, 7], shape=[-1, 1])

        PNNX = torch.matmul(UPNN, self.mat2PX)
        PNNY = torch.matmul(UPNN, self.mat2PY)
        PNNZ = torch.matmul(UPNN, self.mat2PZ)
        PNNR = torch.matmul(UPNN, self.mat2PR)
        PNNS = torch.matmul(UPNN, self.mat2PS)
        PNNT = torch.matmul(UPNN, self.mat2PT)
        PNNV = torch.matmul(UPNN, self.mat2PV)
        PNNW = torch.matmul(UPNN, self.mat2PW)

        gradPNNX = torch.autograd.grad(PNNX, XYZRSTVW, grad_outputs=torch.ones_like(X), create_graph=True,
                                       retain_graph=True)[0]
        gradPNNY = torch.autograd.grad(PNNY, XYZRSTVW, grad_outputs=torch.ones_like(X), create_graph=True,
                                       retain_graph=True)[0]
        gradPNNZ = torch.autograd.grad(PNNZ, XYZRSTVW, grad_outputs=torch.ones_like(X), create_graph=True,
                                       retain_graph=True)[0]
        gradPNNR = torch.autograd.grad(PNNR, XYZRSTVW, grad_outputs=torch.ones_like(X), create_graph=True,
                                       retain_graph=True)[0]
        gradPNNS = torch.autograd.grad(PNNS, XYZRSTVW, grad_outputs=torch.ones_like(X), create_graph=True,
                                       retain_graph=True)[0]
        gradPNNT = torch.autograd.grad(PNNT, XYZRSTVW, grad_outputs=torch.ones_like(X), create_graph=True,
                                       retain_graph=True)[0]
        gradPNNV = torch.autograd.grad(PNNV, XYZRSTVW, grad_outputs=torch.ones_like(X), create_graph=True,
                                       retain_graph=True)[0]
        gradPNNW = torch.autograd.grad(PNNW, XYZRSTVW, grad_outputs=torch.ones_like(X), create_graph=True,
                                       retain_graph=True)[0]

        dPNNX2dx = torch.reshape(gradPNNX[:, 0], shape=[-1, 1])
        dPNNY2dy = torch.reshape(gradPNNY[:, 1], shape=[-1, 1])
        dPNNZ2dz = torch.reshape(gradPNNZ[:, 2], shape=[-1, 1])

        dPNNR2dr = torch.reshape(gradPNNR[:, 3], shape=[-1, 1])
        dPNNS2ds = torch.reshape(gradPNNS[:, 4], shape=[-1, 1])
        dPNNT2dt = torch.reshape(gradPNNT[:, 5], shape=[-1, 1])

        dPNNV2dv = torch.reshape(gradPNNV[:, 6], shape=[-1, 1])
        dPNNW2dw = torch.reshape(gradPNNW[:, 7], shape=[-1, 1])

        if if_lambda2aeps:
            aeps_side = aeps(X, Y, Z, R, S, T, V, W)
        else:
            aeps_side = aeps

        loss2AdUx_temp = PNNX - torch.multiply(aeps_side, dUNN2dx)
        loss2AdUy_temp = PNNY - torch.multiply(aeps_side, dUNN2dy)
        loss2AdUz_temp = PNNZ - torch.multiply(aeps_side, dUNN2dz)

        loss2AdUr_temp = PNNR - torch.multiply(aeps_side, dUNN2dr)
        loss2AdUs_temp = PNNS - torch.multiply(aeps_side, dUNN2ds)
        loss2AdUt_temp = PNNT - torch.multiply(aeps_side, dUNN2dt)

        loss2AdUv_temp = PNNV - torch.multiply(aeps_side, dUNN2dv)
        loss2AdUw_temp = PNNW - torch.multiply(aeps_side, dUNN2dw)

        # loss2func_temp = -(dPNNX2dx + dPNNY2dy + dPNNZ2dz) - force_side
        loss2func_temp = dPNNX2dx + dPNNY2dy + dPNNZ2dz + dPNNR2dr + \
                         dPNNS2ds + dPNNT2dt + dPNNV2dv + dPNNW2dw + force_side
        # -div(a·grad U) = f, in Omega --> (P1, P2,...) = a·grad U --> -div(P) = f --> -(P1x+P2y+P3z+...) = f
        # Vec(P) = (P1, P2,...) = a·grad U =(aUx, aUy, aUz,....)

        if str.lower(loss_type) == 'l2_loss':
            square_loss2func = torch.square(loss2func_temp)
            square_loss2dAU = torch.square(loss2AdUx_temp) + torch.square(loss2AdUy_temp) + \
                              torch.square(loss2AdUz_temp) + torch.square(loss2AdUr_temp) + \
                              torch.square(loss2AdUs_temp) + torch.square(loss2AdUt_temp) + \
                              torch.square(loss2AdUv_temp) + torch.square(loss2AdUw_temp)

            loss2func = torch.mean(square_loss2func)
            loss2dAU = torch.mean(square_loss2dAU)
        elif str.lower(loss_type) == 'lncosh_loss':
            lncosh_loss2func = (1 / scale2lncosh) * torch.log(torch.cosh(scale2lncosh * loss2func_temp))
            lncosh_loss2dAUx = (1 / scale2lncosh) * torch.log(torch.cosh(scale2lncosh * loss2AdUx_temp))
            lncosh_loss2dAUy = (1 / scale2lncosh) * torch.log(torch.cosh(scale2lncosh * loss2AdUy_temp))
            lncosh_loss2dAUz = (1 / scale2lncosh) * torch.log(torch.cosh(scale2lncosh * loss2AdUz_temp))
            lncosh_loss2dAUr = (1 / scale2lncosh) * torch.log(torch.cosh(scale2lncosh * loss2AdUr_temp))
            lncosh_loss2dAUs = (1 / scale2lncosh) * torch.log(torch.cosh(scale2lncosh * loss2AdUs_temp))
            lncosh_loss2dAUt = (1 / scale2lncosh) * torch.log(torch.cosh(scale2lncosh * loss2AdUt_temp))
            lncosh_loss2dAUv = (1 / scale2lncosh) * torch.log(torch.cosh(scale2lncosh * loss2AdUv_temp))
            lncosh_loss2dAUw = (1 / scale2lncosh) * torch.log(torch.cosh(scale2lncosh * loss2AdUw_temp))
            loss2func = torch.mean(lncosh_loss2func)
            loss2dAU = torch.mean(lncosh_loss2dAUx)+torch.mean(lncosh_loss2dAUy)+torch.mean(lncosh_loss2dAUz) + \
                       torch.mean(lncosh_loss2dAUr)+torch.mean(lncosh_loss2dAUs)+torch.mean(lncosh_loss2dAUt) + \
                       torch.mean(lncosh_loss2dAUv)+torch.mean(lncosh_loss2dAUw)
        return UNN, loss2func, loss2dAU

    def loss2bd(self, XYZRSTVW_bd=None, Ubd_exact=None, if_lambda2Ubd=True, loss_type='l2_loss', scale2lncosh=0.1):
        """
        Calculating the loss of Laplace equation (*) on the boundary points for given boundary
        -Laplace U = f,  in Omega
        U = g            on Partial Omega
        Args:
            XYZRSTVW_bd:      the input data of variable. ------  float, shape=[B,D]
            Ubd_exact:     the exact function or array for boundary condition
            if_lambda2Ubd: the Ubd_exact is a lambda function or an array  ------ Bool
            loss_type:     the type of loss function(Ritz, L2, Lncosh)      ------ string
            scale2lncosh:  if the loss is lncosh, using it                  ------- float
        return:
            loss_bd: the output loss on the boundary points for given boundary
        """
        assert (XYZRSTVW_bd is not None)
        assert (Ubd_exact is not None)

        shape2XYZRSTVW = XYZRSTVW_bd.shape
        lenght2XYZRSTVW_shape = len(shape2XYZRSTVW)
        assert (lenght2XYZRSTVW_shape == 2)
        assert (shape2XYZRSTVW[-1] == 8)

        X_bd = torch.reshape(XYZRSTVW_bd[:, 0], shape=[-1, 1])
        Y_bd = torch.reshape(XYZRSTVW_bd[:, 1], shape=[-1, 1])
        Z_bd = torch.reshape(XYZRSTVW_bd[:, 2], shape=[-1, 1])
        R_bd = torch.reshape(XYZRSTVW_bd[:, 3], shape=[-1, 1])
        S_bd = torch.reshape(XYZRSTVW_bd[:, 4], shape=[-1, 1])
        T_bd = torch.reshape(XYZRSTVW_bd[:, 5], shape=[-1, 1])
        V_bd = torch.reshape(XYZRSTVW_bd[:, 6], shape=[-1, 1])
        W_bd = torch.reshape(XYZRSTVW_bd[:, 7], shape=[-1, 1])

        if if_lambda2Ubd:
            Ubd = Ubd_exact(X_bd, Y_bd, Z_bd, R_bd, S_bd, T_bd, V_bd, W_bd)
        else:
            Ubd = Ubd_exact

        UPNN_bd = self.DNN(XYZRSTVW_bd, scale=self.factor2freq, sFourier=self.sFourier)
        UNN_bd = torch.matmul(UPNN_bd, self.mat2U)
        diff2bd = UNN_bd - Ubd
        if str.lower(loss_type) == 'l2_loss':
            loss_bd_square = torch.mul(diff2bd, diff2bd)
            loss_bd = torch.mean(loss_bd_square)
        elif str.lower(loss_type) == 'lncosh_loss':
            loss_bd_temp = (1 / scale2lncosh) * torch.log(torch.cosh(scale2lncosh * diff2bd))
            loss_bd = torch.mean(loss_bd_temp)
        return loss_bd

    def get_regularSum2WB(self):
        """
        Calculating the regularization sum of weights and biases
        """
        sum2WB = self.DNN.get_regular_sum2WB(regular_model=self.opt2regular_WB)
        return sum2WB

    def eval_FMPINN(self, XYZRSTVW_points=None):
        """
        Evaluating the FMPINN for testing points
        Args:
            XYZRSTVW_points: the testing input data of variable. ------  float, shape=[B,D]
        return:
            UNN: the testing output
        """
        assert (XYZRSTVW_points is not None)
        shape2XYZRSTVW = XYZRSTVW_points.shape
        lenght2XYZRSTVW_shape = len(shape2XYZRSTVW)
        assert (lenght2XYZRSTVW_shape == 2)
        assert (shape2XYZRSTVW[-1] == 8)

        UPNN = self.DNN(XYZRSTVW_points, scale=self.factor2freq, sFourier=self.sFourier)
        UNN = torch.matmul(UPNN, self.mat2U)
        return UNN


def solve_Multiscale_PDE(R):
    log_out_path = R['FolderName']        # 将路径从字典 R 中提取出来
    if not os.path.exists(log_out_path):  # 判断路径是否已经存在
        os.mkdir(log_out_path)            # 无 log_out_path 路径，创建一个 log_out_path 路径
    logfile_name = '%s%s.txt' % ('log2', R['name2act_hidden'])
    log_fileout = open(os.path.join(log_out_path, logfile_name), 'w')  # 在这个路径下创建并打开一个可写的 log_train.txt文件
    DNN_Log_Print.dictionary_out2file(R, log_fileout)

    DNN_tools.log_string('The penalty of flux(dual) variable": %s\n\n' % (R['gradient_penalty']), log_fileout)

    # 一般 laplace 问题需要的设置
    batchsize_it = R['batch_size2interior']
    batchsize_bd = R['batch_size2boundary']

    bd_penalty_init = R['init_boundary_penalty']                # Regularization parameter for boundary conditions
    penalty2WB = R['penalty2weight_biases']                # Regularization parameter for weights and biases
    learning_rate = R['learning_rate']

    penalty2gd = R['gradient_penalty']

    input_dim = R['input_dim']
    out_dim = R['output_dim']

    # pLaplace 算子需要的额外设置, 先预设一下
    region_lb = 0.0
    region_rt = 1.0
    u_true, f, Aeps = MS_LaplaceEqs.get_infos2pLaplace_8D(
        input_dim=input_dim, out_dim=out_dim, intervalL=region_lb, intervalR=region_rt, equa_name=R['equa_name'])

    model = FMPINN(input_dim=R['input_dim'], out_dim=R['output_dim'], hidden_layer=R['hidden_layers'],
                   Model_name=R['model2NN'], name2actIn=R['name2act_in'], name2actHidden=R['name2act_hidden'],
                   name2actOut=R['name2act_out'], opt2regular_WB='L0', type2numeric='float32',
                   factor2freq=R['freq'], use_gpu=R['use_gpu'], No2GPU=R['gpuNo'])

    if True == R['use_gpu']:
        model = model.cuda(device='cuda:' + str(R['gpuNo']))

    params2Net = model.DNN.parameters()

    # 定义优化方法，并给定初始学习率
    # optimizer = torch.optim.SGD(params2Net, lr=init_lr)                     # SGD
    # optimizer = torch.optim.SGD(params2Net, lr=init_lr, momentum=0.8)       # momentum
    # optimizer = torch.optim.RMSprop(params2Net, lr=init_lr, alpha=0.95)     # RMSProp
    optimizer = torch.optim.Adam(params2Net, lr=learning_rate)  # Adam

    # 定义更新学习率的方法
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1/(epoch+1))
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.995)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 100, gamma=0.975)

    t0 = time.time()
    loss_it_all, loss_bd_all, loss_all, train_mse_all, train_rel_all = [], [], [], [], []  # 空列表, 使用 append() 添加元素
    test_mse_all, test_rel_all = [], []
    test_epoch = []
    loss_adu_all = []

    # 画网格解图
    if R['testData_model'] == 'random_generate':
        # 生成测试数据，用于测试训练后的网络
        # test_bach_size = 400
        # size2test = 20
        # test_bach_size = 900
        # size2test = 30
        test_bach_size = 1600
        size2test = 40
        # test_bach_size = 4900
        # size2test = 70
        # test_bach_size = 10000
        # size2test = 100
        # test_bach_size = 40000
        # size2test = 200
        # test_bach_size = 250000
        # size2test = 500
        # test_bach_size = 1000000
        # size2test = 1000
        test_xyzrstvw_torch = dataUtilizer2torch.rand_it(
            test_bach_size, input_dim, region_a=region_lb, region_b=region_rt, to_torch=True, to_float=True,
            to_cuda=R['use_gpu'], gpu_no=0, use_grad2x=False)
    elif R['testData_model'] == 'loadData':
        test_bach_size = 1600
        size2test = 40
        mat_data_path = 'dataMat_highDim'
        test_xyzrstvw_torch = Load_data2Mat.get_randomData2mat(
            dim=input_dim, data_path=mat_data_path, to_torch=True, to_float=True, to_cuda=R['use_gpu'], gpu_no=0,
            use_grad2x=False)

    if True == R['use_gpu']:
        numpy_test_xyzrstvw = test_xyzrstvw_torch.cpu().detach().numpy()
        saveData.save_testData_or_solus2mat(numpy_test_xyzrstvw, dataName='testXYZRSTVW', outPath=R['FolderName'])
    else:
        numpy_test_xyzrstvw = test_xyzrstvw_torch.detach().numpy()
        saveData.save_testData_or_solus2mat(numpy_test_xyzrstvw, dataName='testXYZRSTVW', outPath=R['FolderName'])

    Utrue2test = u_true(torch.reshape(test_xyzrstvw_torch[:, 0], shape=[-1, 1]),
                        torch.reshape(test_xyzrstvw_torch[:, 1], shape=[-1, 1]),
                        torch.reshape(test_xyzrstvw_torch[:, 2], shape=[-1, 1]),
                        torch.reshape(test_xyzrstvw_torch[:, 3], shape=[-1, 1]),
                        torch.reshape(test_xyzrstvw_torch[:, 4], shape=[-1, 1]),
                        torch.reshape(test_xyzrstvw_torch[:, 5], shape=[-1, 1]),
                        torch.reshape(test_xyzrstvw_torch[:, 6], shape=[-1, 1]),
                        torch.reshape(test_xyzrstvw_torch[:, 7], shape=[-1, 1]))

    if True == R['use_gpu']:
        ubd = torch.tensor([0], dtype=torch.float32, device='cuda:' + str(R['gpuNo']))
    else:
        ubd = torch.tensor([0], dtype=torch.float32)

    for i_epoch in range(R['max_epoch'] + 1):
        xyzrstvw_it_batch = dataUtilizer2torch.rand_it_lhs(batchsize_it, R['input_dim'], region_a=region_lb,
                                                           region_b=region_rt, to_float=True, to_cuda=R['use_gpu'],
                                                           gpu_no=R['gpuNo'], use_grad2x=True)
        xyzrstvw00_batch, xyzrstvw01_batch, xyzrstvw10_batch, xyzrstvw11_batch, xyzrstvw20_batch, xyzrstvw21_batch, \
        xyzrstvw30_batch, xyzrstvw31_batch, xyzrstvw40_batch, xyzrstvw41_batch, xyzrstvw50_batch, xyzrstvw51_batch, \
        xyzrstvw60_batch, xyzrstvw61_batch, xyzrstvw70_batch, xyzrstvw71_batch = dataUtilizer2torch.rand_bd_8D_lhs(
            batchsize_bd, R['input_dim'], region_a=region_lb, region_b=region_rt, to_float=True, to_cuda=R['use_gpu'],
            gpu_no=R['gpuNo'])

        if R['activate_penalty2bd_increase'] == 1:
            if i_epoch < int(R['max_epoch'] / 10):
                temp_penalty_bd = bd_penalty_init
            elif i_epoch < int(R['max_epoch'] / 5):
                temp_penalty_bd = 10 * bd_penalty_init
            elif i_epoch < int(R['max_epoch'] / 4):
                temp_penalty_bd = 50 * bd_penalty_init
            elif i_epoch < int(R['max_epoch'] / 2):
                temp_penalty_bd = 100 * bd_penalty_init
            elif i_epoch < int(3 * R['max_epoch'] / 4):
                temp_penalty_bd = 200 * bd_penalty_init
            else:
                temp_penalty_bd = 500 * bd_penalty_init
        else:
            temp_penalty_bd = bd_penalty_init

        X = torch.reshape(xyzrstvw_it_batch[:, 0], shape=[-1, 1])
        Y = torch.reshape(xyzrstvw_it_batch[:, 1], shape=[-1, 1])
        Z = torch.reshape(xyzrstvw_it_batch[:, 2], shape=[-1, 1])
        RRR = torch.reshape(xyzrstvw_it_batch[:, 3], shape=[-1, 1])
        S = torch.reshape(xyzrstvw_it_batch[:, 4], shape=[-1, 1])
        T = torch.reshape(xyzrstvw_it_batch[:, 5], shape=[-1, 1])
        V = torch.reshape(xyzrstvw_it_batch[:, 6], shape=[-1, 1])
        W = torch.reshape(xyzrstvw_it_batch[:, 7], shape=[-1, 1])
        force = MS_LaplaceEqs.get_infos2force_8D(x=X, y=Y, z=Z, r=RRR, s=S, t=T, v=V, w=W)
        UNN2train, loss_it, loss_dAU = model.loss_in2pLaplace(
            XYZRSTVW=xyzrstvw_it_batch, fside=force, if_lambda2fside=False, aeps=Aeps, if_lambda2aeps=True,
            loss_type=R['loss_type'], scale2lncosh=0.1)

        loss_bd00 = model.loss2bd(XYZRSTVW_bd=xyzrstvw00_batch, Ubd_exact=ubd,
                                  if_lambda2Ubd=False, loss_type=R['loss_type2bd'])
        loss_bd01 = model.loss2bd(XYZRSTVW_bd=xyzrstvw01_batch, Ubd_exact=ubd,
                                  if_lambda2Ubd=False, loss_type=R['loss_type2bd'])
        loss_bd10 = model.loss2bd(XYZRSTVW_bd=xyzrstvw10_batch, Ubd_exact=ubd,
                                  if_lambda2Ubd=False, loss_type=R['loss_type2bd'])
        loss_bd11 = model.loss2bd(XYZRSTVW_bd=xyzrstvw11_batch, Ubd_exact=ubd,
                                  if_lambda2Ubd=False, loss_type=R['loss_type2bd'])
        loss_bd20 = model.loss2bd(XYZRSTVW_bd=xyzrstvw20_batch, Ubd_exact=ubd,
                                  if_lambda2Ubd=False, loss_type=R['loss_type2bd'])
        loss_bd21 = model.loss2bd(XYZRSTVW_bd=xyzrstvw21_batch, Ubd_exact=ubd,
                                  if_lambda2Ubd=False, loss_type=R['loss_type2bd'])
        loss_bd30 = model.loss2bd(XYZRSTVW_bd=xyzrstvw30_batch, Ubd_exact=ubd,
                                  if_lambda2Ubd=False, loss_type=R['loss_type2bd'])
        loss_bd31 = model.loss2bd(XYZRSTVW_bd=xyzrstvw31_batch, Ubd_exact=ubd,
                                  if_lambda2Ubd=False, loss_type=R['loss_type2bd'])
        loss_bd40 = model.loss2bd(XYZRSTVW_bd=xyzrstvw40_batch, Ubd_exact=ubd,
                                  if_lambda2Ubd=False, loss_type=R['loss_type2bd'])
        loss_bd41 = model.loss2bd(XYZRSTVW_bd=xyzrstvw41_batch, Ubd_exact=ubd,
                                  if_lambda2Ubd=False, loss_type=R['loss_type2bd'])
        loss_bd50 = model.loss2bd(XYZRSTVW_bd=xyzrstvw50_batch, Ubd_exact=ubd,
                                  if_lambda2Ubd=False, loss_type=R['loss_type2bd'])
        loss_bd51 = model.loss2bd(XYZRSTVW_bd=xyzrstvw51_batch, Ubd_exact=ubd,
                                  if_lambda2Ubd=False, loss_type=R['loss_type2bd'])
        loss_bd60 = model.loss2bd(XYZRSTVW_bd=xyzrstvw60_batch, Ubd_exact=ubd,
                                  if_lambda2Ubd=False, loss_type=R['loss_type2bd'])
        loss_bd61 = model.loss2bd(XYZRSTVW_bd=xyzrstvw61_batch, Ubd_exact=ubd,
                                  if_lambda2Ubd=False, loss_type=R['loss_type2bd'])
        loss_bd70 = model.loss2bd(XYZRSTVW_bd=xyzrstvw70_batch, Ubd_exact=ubd,
                                  if_lambda2Ubd=False, loss_type=R['loss_type2bd'])
        loss_bd71 = model.loss2bd(XYZRSTVW_bd=xyzrstvw71_batch, Ubd_exact=ubd,
                                  if_lambda2Ubd=False, loss_type=R['loss_type2bd'])

        loss_bd = loss_bd00 + loss_bd01 + loss_bd10 + loss_bd11 + loss_bd20 + loss_bd21 + loss_bd30 + loss_bd31 + \
                  loss_bd40 + loss_bd41 + loss_bd50 + loss_bd51 + loss_bd60 + loss_bd61 + loss_bd70 + loss_bd71

        PWB = penalty2WB * model.get_regularSum2WB()

        loss = loss_it + penalty2gd * loss_dAU + temp_penalty_bd * loss_bd + PWB  # 要优化的loss function

        loss_it_all.append(loss_it.item())
        loss_bd_all.append(loss_bd.item())
        loss_adu_all.append(loss_dAU.item())
        loss_all.append(loss.item())

        optimizer.zero_grad()                 # 求导前先清零, 只要在下一次求导前清零即可
        loss.backward()                       # 对loss关于Ws和Bs求偏导
        optimizer.step()                      # 更新参数Ws和Bs
        scheduler.step()

        Uexact2train = u_true(X, Y, Z, RRR, S, T, V, W)
        train_mse = torch.mean(torch.square(UNN2train - Uexact2train))
        train_rel = train_mse / torch.mean(torch.square(Uexact2train))

        train_mse_all.append(train_mse.item())
        train_rel_all.append(train_rel.item())

        # if i_epoch % 100 == 0:
        if i_epoch % 1000 == 0:
            tmp_lr = optimizer.param_groups[0]['lr']
            run_times = time.time() - t0
            DNN_tools.print_and_log_train_one_epoch(
                i_epoch, run_times, tmp_lr, temp_penalty_bd, PWB, loss_it.item(), loss_dAU.item(), loss_bd.item(),
                loss.item(), train_mse.item(), train_rel.item(), log_out=log_fileout)

            # ---------------------------   test network ----------------------------------------------
            test_epoch.append(i_epoch / 1000)
            UNN2test = model.eval_FMPINN(XYZRSTVW_points=test_xyzrstvw_torch)

            point_square_error = torch.square(Utrue2test - UNN2test)
            test_mse = torch.mean(point_square_error)
            test_rel = test_mse / torch.mean(torch.square(Utrue2test))
            test_mse_all.append(test_mse.item())
            test_rel_all.append(test_rel.item())
            DNN_tools.print_and_log_test_one_epoch(test_mse.item(), test_rel.item(), log_out=log_fileout)

    # ------------------- save the training results into mat file and plot them -------------------------
    saveData.save_trainLoss2mat_1actFunc(loss_it_all, loss_bd_all, loss_all, actName=R['name2act_hidden'],
                                         outPath=R['FolderName'])
    saveData.save_trainLoss2mat(loss_adu_all, lossName='Loss_Adu', actName=R['name2act_hidden'],
                                outPath=R['FolderName'])

    saveData.save_train_MSE_REL2mat(train_mse_all, train_rel_all, actName=R['name2act_hidden'],
                                    outPath=R['FolderName'])

    plotData.plotTrain_loss_1act_func(loss_it_all, lossType='loss_it', seedNo=R['seed'], outPath=R['FolderName'])
    plotData.plotTrain_loss_1act_func(loss_bd_all, lossType='loss_bd', seedNo=R['seed'], outPath=R['FolderName'],
                                      yaxis_scale=True)
    plotData.plotTrain_loss_1act_func(loss_all, lossType='loss', seedNo=R['seed'], outPath=R['FolderName'])

    saveData.save_train_MSE_REL2mat(train_mse_all, train_rel_all, actName=R['name2act_hidden'],
                                    outPath=R['FolderName'])
    plotData.plotTrain_MSE_REL_1act_func(train_mse_all, train_rel_all, actName=R['name2act_hidden'], seedNo=R['seed'],
                                         outPath=R['FolderName'], yaxis_scale=True)

    # ----------------------  save testing results to mat files, then plot them --------------------------------
    if True == R['use_gpu']:
        utrue2test_numpy = Utrue2test.cpu().detach().numpy()
        unn2test_numpy = UNN2test.cpu().detach().numpy()
        point_square_error_numpy = point_square_error.cpu().detach().numpy()
    else:
        utrue2test_numpy = Utrue2test.detach().numpy()
        unn2test_numpy = UNN2test.detach().numpy()
        point_square_error_numpy = point_square_error.detach().numpy()

    saveData.save_2testSolus2mat(utrue2test_numpy, unn2test_numpy, actName='utrue', actName1=R['name2act_hidden'],
                                 outPath=R['FolderName'])

    saveData.save_testMSE_REL2mat(test_mse_all, test_rel_all, actName=R['name2act_hidden'], outPath=R['FolderName'])
    saveData.save_test_point_wise_err2mat(point_square_error_numpy, actName=R['name2act_hidden'], outPath=R['FolderName'])

    plotData.plotTest_MSE_REL(test_mse_all, test_rel_all, test_epoch, actName=R['name2act_hidden'],
                              seedNo=R['seed'], outPath=R['FolderName'], yaxis_scale=True)

    plotData.plot_Hot_point_wise_err(point_square_error_numpy, size_vec2mat=size2test, actName=R['name2act_hidden'],
                                     seedNo=R['seed'], outPath=R['FolderName'])
    plotData.plot_Hot_solution2test(utrue2test_numpy, size_vec2mat=size2test, actName='Utrue', seedNo=R['seed'],
                                    outPath=R['FolderName'])
    plotData.plot_Hot_solution2test(unn2test_numpy, size_vec2mat=size2test, actName=R['name2act_hidden'],
                                    seedNo=R['seed'], outPath=R['FolderName'])


if __name__ == "__main__":
    R={}
    # -------------------------------------- CPU or GPU 选择 -----------------------------------------------
    R['gpuNo'] = 0
    # 默认使用 GPU，这个标记就不要设为-1，设为0,1,2,3,4....n（n指GPU的数目，即电脑有多少块GPU）
    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # -1代表使用 CPU 模式
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 设置当前使用的GPU设备仅为第 0 块GPU, 设备名称为'/gpu:0'
    if platform.system() == 'Windows':
        os.environ["CDUA_VISIBLE_DEVICES"] = "%s" % (R['gpuNo'])
    else:
        print('-------------------------------------- linux -----------------------------------------------')
        # Linux终端没有GUI, 需要添加如下代码，而且必须添加在 import matplotlib.pyplot 之前，否则无效。
        matplotlib.use('Agg')

    # 文件保存路径设置
    file2results = 'Results'
    store_file = 'FMPINN8D'
    BASE_DIR2FILE = os.path.dirname(os.path.abspath(__file__))
    split_BASE_DIR2FILE = os.path.split(BASE_DIR2FILE)
    split_BASE_DIR2FILE = os.path.split(split_BASE_DIR2FILE[0])
    BASE_DIR = split_BASE_DIR2FILE[0]
    # print(split_BASE_DIR2FILE)
    sys.path.append(BASE_DIR)
    OUT_DIR_BASE = os.path.join(BASE_DIR, file2results)
    OUT_DIR = os.path.join(OUT_DIR_BASE, store_file)
    sys.path.append(OUT_DIR)
    if not os.path.exists(OUT_DIR):
        print('---------------------- OUT_DIR ---------------------:', OUT_DIR)
        os.mkdir(OUT_DIR)

    R['seed'] = np.random.randint(1e5)
    seed_str = str(R['seed'])                     # int 型转为字符串型
    FolderName = os.path.join(OUT_DIR, seed_str)  # 路径连接
    R['FolderName'] = FolderName
    if not os.path.exists(FolderName):
        print('--------------------- FolderName -----------------:', FolderName)
        os.mkdir(FolderName)

    # ----------------------------------------  复制并保存当前文件 -----------------------------------------
    if platform.system() == 'Windows':
        shutil.copy(__file__, '%s/%s' % (FolderName, os.path.basename(__file__)))
    else:
        shutil.copy(__file__, '%s/%s' % (FolderName, os.path.basename(__file__)))

    # if the value of step_stop_flag is not 0, it will activate stop condition of step to kill program
    step_stop_flag = input('please input an  integer number to activate step-stop----0:no---!0:yes--:')
    R['activate_stop'] = int(step_stop_flag)
    # if the value of step_stop_flag is not 0, it will activate stop condition of step to kill program
    R['max_epoch'] = 50000
    if 0 != R['activate_stop']:
        epoch_stop = input('please input a stop epoch:')
        R['max_epoch'] = int(epoch_stop)

    # ---------------------------- Setup of multi-scale problem-------------------------------
    R['input_dim'] = 8     # 输入维数，即问题的维数(几元问题)
    R['output_dim'] = 1+8  # 输出维数

    R['PDE_type'] = 'pLaplace'
    R['equa_name'] = 'multi_scale8D_1'
    # R['equa_name'] = 'multi_scale8D_2'

    R['mesh_number'] = 1
    R['epsilon'] = 0.1
    R['order2pLaplace_operator'] = 2
    # R['batch_size2interior'] = 10000  # 内部训练数据的批大小
    R['batch_size2interior'] = 12500  # 内部训练数据的批大小
    # R['batch_size2interior'] = 20000  # 内部训练数据的批大小
    # R['batch_size2boundary'] = 5000
    R['batch_size2boundary'] = 2500

    # ---------------------------- Setup of DNN -------------------------------
    # 装载测试数据模式
    # R['testData_model'] = 'loadData'
    R['testData_model'] = 'random_generate'

    R['loss_type'] = 'L2_loss'                        # loss类型:L2 loss
    # R['loss_type'] = 'lncosh_loss'                        # loss类型:L2 loss

    # R['scale2lncosh'] = 0.01
    R['scale2lncosh'] = 0.05
    # R['scale2lncosh'] = 0.1
    # R['scale2lncosh'] = 0.5
    # R['scale2lncosh'] = 1

    R['loss_type2bd'] = 'l2_loss'

    if R['loss_type'] == 'lncosh_loss':
        R['loss_type2bd'] = 'lncosh_loss'

    R['optimizer_name'] = 'Adam'                        # 优化器
    # R['learning_rate'] = 2e-4                           # 学习率
    R['learning_rate'] = 0.01  # 学习率  这个学习率最好
    # R['learning_rate'] = 0.005  # 学习率
    R['learning_rate_decay'] = 5e-5                     # 学习率 decay
    R['train_model'] = 'union_training'
    # R['train_model'] = 'group2_training'
    # R['train_model'] = 'group3_training'

    # 正则化权重和偏置的模式
    R['regular_wb_model'] = 'L0'
    # R['regular_wb_model'] = 'L1'
    # R['regular_wb_model'] = 'L2'
    R['penalty2weight_biases'] = 0.000                      # Regularization parameter for weights
    # R['penalty2weight_biases'] = 0.001                    # Regularization parameter for weights
    # R['penalty2weight_biases'] = 0.0025                   # Regularization parameter for weights

    # 边界的惩罚处理方式,以及边界的惩罚因子
    R['activate_penalty2bd_increase'] = 1
    # R['init_boundary_penalty'] = 1000  # Regularization parameter for boundary conditions
    R['init_boundary_penalty'] = 10  # Regularization parameter for boundary conditions

    R['gradient_penalty'] = 10  # Regularization parameter for boundary conditions

    # &&&&&&&&&&&&&&&&&&& 使用的网络模型 &&&&&&&&&&&&&&&&&&&&&&&&&&&
    R['model2NN'] = 'Fourier_SubDNN'

    # &&&&&&&&&&&&&&&&&&&&&& 隐藏层的层数和每层神经元数目 &&&&&&&&&&&&&&&&&&&&&&&&&&&&
    # R['hidden_layers'] = (5, 8, 6, 6, 4)
    # R['freq'] = np.arange(1, 51)

    # R['hidden_layers'] = (5, 10, 6, 6, 5)
    # R['freq'] = np.concatenate(([1], np.arange(start=2, stop=51, step=2)), axis=0)

    # R['hidden_layers'] = (5, 12, 8, 8, 6)
    # R['freq'] = np.concatenate(([1], np.arange(start=2, stop=51, step=2)), axis=0)

    # R['hidden_layers'] = (10, 15, 8, 8, 6)
    # R['freq'] = np.concatenate(([1], np.arange(start=2, stop=51, step=2)), axis=0)

    # R['hidden_layers'] = (15, 20, 10, 10, 6)
    # R['freq'] = np.concatenate(([1], np.arange(start=2, stop=51, step=2)), axis=0)

    # R['hidden_layers'] = (20, 40, 30, 30, 15)
    # R['freq'] = np.concatenate(([1], np.arange(start=2, stop=51, step=2)), axis=0)

    # R['hidden_layers'] = (40, 60, 40, 40, 20)
    # R['freq'] = np.concatenate(([1], np.arange(start=2, stop=51, step=2)), axis=0)

    # R['hidden_layers'] = (40, 60, 40, 40, 40)
    # R['hidden_layers'] = (50, 80, 50, 50, 50)
    R['hidden_layers'] = (60, 80, 60, 60, 60)
    # R['freq'] = np.concatenate(([1], np.arange(start=2, stop=51, step=2)), axis=0)
    # R['freq'] = np.concatenate(([1], np.arange(start=2, stop=65, step=3)), axis=0)
    # R['freq'] = np.concatenate(([1], np.arange(start=2, stop=111, step=5)), axis=0)
    # R['freq'] = np.concatenate(([1], np.arange(start=2, stop=100, step=4)), axis=0)
    R['freq'] = np.concatenate(([1, 2, 3, 4, 5], np.arange(start=10, stop=100, step=5), [100]), axis=0)
    # R['freq'] = np.array([1, 2, 4, 8, 16, 32, 64])

    # &&&&&&&&&&&&&&&&&&& 激活函数的选择 &&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    # R['name2act_in'] = 'relu'
    # R['name2act_in'] = 'tanh'
    # R['name2act_in'] = 'sin'
    # R['name2act_in'] = 'enhance_tanh'
    # R['name2act_in'] = 's2relu'
    R['name2act_in'] = 'sinAddcos'

    # R['name2act_hidden'] = 'relu'
    # R['name2act_hidden'] = 'tanh'
    # R['name2act_hidden'] = 'enhance_tanh'
    # R['name2act_hidden']' = leaky_relu'
    # R['name2act_hidden'] = 'srelu'
    # R['name2act_hidden'] = 's2relu'
    # R['name2act_hidden'] = 'sin'
    R['name2act_hidden'] = 'sinAddcos'
    # R['name2act_hidden'] = 'elu'
    # R['name2act_hidden'] = 'phi'

    R['name2act_out'] = 'linear'

    R['sfourier'] = 1.0
    # R['sfourier'] = 5.0
    # R['sfourier'] = 0.75

    # R['use_gpu'] = False
    R['use_gpu'] = True

    R['repeat_High_freq'] = True

    solve_Multiscale_PDE(R)

