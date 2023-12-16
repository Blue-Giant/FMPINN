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


class MscaleDNN(tn.Module):
    def __init__(self, input_dim=4, out_dim=1, hidden_layer=None, Model_name='DNN', name2actIn='relu',
                 name2actHidden='relu', name2actOut='linear', opt2regular_WB='L2', type2numeric='float32',
                 factor2freq=None, sFourier=1.0, repeat_highFreq=True, use_gpu=False, No2GPU=0):
        """
        initialing the class of MscaleDNN with given setups
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
        super(MscaleDNN, self).__init__()
        if Model_name=='FOURIER_SUBDNN':
            self.DNN2Solu = DNN_base.Fourier_SubNets3D(
                indim=input_dim, outdim=out_dim, hidden_units=hidden_layer, name2Model=Model_name,
                actName2in=name2actIn, actName=name2actHidden, actName2out=name2actOut, type2float=type2numeric,
                repeat_Highfreq=repeat_highFreq, to_gpu=use_gpu, gpu_no=No2GPU, num2subnets=len(factor2freq))

            self.DNN2PNNX = DNN_base.Fourier_SubNets3D(
                indim=input_dim, outdim=out_dim, hidden_units=hidden_layer, name2Model=Model_name,
                actName2in=name2actIn, actName=name2actHidden, actName2out=name2actOut, type2float=type2numeric,
                repeat_Highfreq=repeat_highFreq, to_gpu=use_gpu, gpu_no=No2GPU, num2subnets=len(factor2freq))

            self.DNN2PNNY = DNN_base.Fourier_SubNets3D(
                indim=input_dim, outdim=out_dim, hidden_units=hidden_layer, name2Model=Model_name,
                actName2in=name2actIn, actName=name2actHidden, actName2out=name2actOut, type2float=type2numeric,
                repeat_Highfreq=repeat_highFreq, to_gpu=use_gpu, gpu_no=No2GPU, num2subnets=len(factor2freq))
        else:
            self.DNN2Solu = DNN_base.Dense_FourierNet(
                indim=input_dim, outdim=out_dim, hidden_units=hidden_layer, name2Model=Model_name,
                actName2in=name2actIn, actName=name2actHidden, actName2out=name2actOut, type2float=type2numeric,
                repeat_Highfreq=repeat_highFreq, to_gpu=use_gpu, gpu_no=No2GPU)

            self.DNN2PNNX = DNN_base.Dense_FourierNet(
                indim=input_dim, outdim=out_dim, hidden_units=hidden_layer, name2Model=Model_name,
                actName2in=name2actIn, actName=name2actHidden, actName2out=name2actOut, type2float=type2numeric,
                repeat_Highfreq=repeat_highFreq, to_gpu=use_gpu, gpu_no=No2GPU)

            self.DNN2PNNY = DNN_base.Dense_FourierNet(
                indim=input_dim, outdim=out_dim, hidden_units=hidden_layer, name2Model=Model_name,
                actName2in=name2actIn, actName=name2actHidden, actName2out=name2actOut, type2float=type2numeric,
                repeat_Highfreq=repeat_highFreq, to_gpu=use_gpu, gpu_no=No2GPU)

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

        self.mat2X = torch.tensor([[1, 0]], dtype=self.float_type,
                                  device=self.opt2device, requires_grad=False)  # 2 行 1 列
        self.mat2Y = torch.tensor([[0, 1]], dtype=self.float_type,
                                  device=self.opt2device, requires_grad=False)  # 2 行 1 列

    def loss_in2pLaplace(self, X=None, Y=None, fside=None, if_lambda2fside=True, aeps=None, if_lambda2aeps=True,
                         loss_type='ritz_loss', scale2lncosh=0.1):
        """
        Calculating the loss of Laplace equation (*) in the interior points for given domain
        -div(a·grad U) = f,  in Omega
        U = g            on Partial Omega
        Args:
             XY:              the input data of variable. ------  float, shape=[B,D]
             fside:           the force side              ------  float, shape=[B,1]
             if_lambda2fside: the force-side is a lambda function or an array  ------ Bool
             loss_type:       the type of loss function(Ritz, L2, Lncosh)      ------ string
             scale2lncosh:    if the loss is lncosh, using it                  ------- float
        return:
             UNN:             the output data
             loss_in:         the output loss in the interior points for given domain
        """
        assert (X is not None)
        assert (Y is not None)
        assert (fside is not None)

        shape2X = X.shape
        lenght2X_shape = len(shape2X)
        assert (lenght2X_shape == 2)
        assert (shape2X[-1] == 1)

        shape2Y = Y.shape
        lenght2Y_shape = len(shape2Y)
        assert (lenght2Y_shape == 2)
        assert (shape2Y[-1] == 1)

        if if_lambda2fside:
            force_side = fside(X, Y)
        else:
            force_side = fside

        XY = torch.matmul(X, self.mat2X) + torch.matmul(Y, self.mat2Y)
        UNN = self.DNN2Solu(XY, scale=self.factor2freq, sFourier=self.sFourier)
        PNNX = self.DNN2PNNX(XY, scale=self.factor2freq, sFourier=self.sFourier)
        PNNY = self.DNN2PNNY(XY, scale=self.factor2freq, sFourier=self.sFourier)

        gradX2UNN = torch.autograd.grad(UNN, X, grad_outputs=torch.ones_like(X), create_graph=True, retain_graph=True)
        gradY2UNN = torch.autograd.grad(UNN, Y, grad_outputs=torch.ones_like(Y), create_graph=True, retain_graph=True)
        dUNN2dx = torch.reshape(gradX2UNN[0], shape=[-1, 1])
        dUNN2dy = torch.reshape(gradY2UNN[0], shape=[-1, 1])

        gradPNNX = torch.autograd.grad(PNNX, X, grad_outputs=torch.ones_like(X), create_graph=True,
                                       retain_graph=True)
        gradPNNY = torch.autograd.grad(PNNY, Y, grad_outputs=torch.ones_like(Y), create_graph=True,
                                       retain_graph=True)

        dPNNX2dx = torch.reshape(gradPNNX[0], shape=[-1, 1])
        dPNNY2dy = torch.reshape(gradPNNY[0], shape=[-1, 1])

        if if_lambda2aeps:
            aeps_side = aeps(X, Y)
        else:
            aeps_side = aeps
        loss2dAUx_temp = PNNX - torch.multiply(aeps_side, dUNN2dx)
        loss2dAUy_temp = PNNY - torch.multiply(aeps_side, dUNN2dy)

        # loss2func_temp = -torch.add(dPNNX2dx, dPNNY2dy) - force_side
        loss2func_temp = torch.add(dPNNX2dx, dPNNY2dy) + force_side
        # -div(a·grad U) = f, in Omega --> (P1, P2,...) = a·grad U --> -div(P) = f --> -(P1x+P2y+P3z+...) = f
        # Vec(P) = (P1, P2,...) = a·grad U =(aUx, aUy, aUz,....)
        if str.lower(loss_type) == 'l2_loss':
            square_loss2func = torch.square(loss2func_temp)
            square_loss2dAU = torch.square(loss2dAUx_temp) + torch.square(loss2dAUy_temp)

            loss2func = torch.mean(square_loss2func)
            loss2dAU = torch.mean(square_loss2dAU)
        elif str.lower(loss_type) == 'lncosh_loss':
            lncosh_loss2func = (1 / scale2lncosh) * torch.log(torch.cosh(scale2lncosh * loss2func_temp))
            lncosh_loss2dAUx = (1 / scale2lncosh) * torch.log(torch.cosh(scale2lncosh * loss2dAUx_temp))
            lncosh_loss2dAUy = (1 / scale2lncosh) * torch.log(torch.cosh(scale2lncosh * loss2dAUy_temp))
            loss2func = torch.mean(lncosh_loss2func)
            loss2dAU = torch.add(torch.mean(lncosh_loss2dAUx), torch.mean(lncosh_loss2dAUy))
        return UNN, loss2func, loss2dAU

    def loss2bd(self, XY_bd=None, Ubd_exact=None, if_lambda2Ubd=True, loss_type='ritz_loss', scale2lncosh=0.1):
        """
        Calculating the loss of Laplace equation (*) on the boundary points for given boundary
        -Laplace U = f,  in Omega
        U = g            on Partial Omega
        Args:
            XY_bd:         the input data of variable. ------  float, shape=[B,D]
            Ubd_exact:     the exact function or array for boundary condition
            if_lambda2Ubd: the Ubd_exact is a lambda function or an array  ------ Bool
            loss_type:     the type of loss function(Ritz, L2, Lncosh)      ------ string
            scale2lncosh:  if the loss is lncosh, using it                  ------- float
        return:
            loss_bd: the output loss on the boundary points for given boundary
        """
        assert (XY_bd is not None)
        assert (Ubd_exact is not None)

        shape2XY = XY_bd.shape
        lenght2XY_shape = len(shape2XY)
        assert (lenght2XY_shape == 2)
        assert (shape2XY[-1] == 2)
        X_bd = torch.reshape(XY_bd[:, 0], shape=[-1, 1])
        Y_bd = torch.reshape(XY_bd[:, 1], shape=[-1, 1])

        if if_lambda2Ubd:
            Ubd = Ubd_exact(X_bd, Y_bd)
        else:
            Ubd = Ubd_exact

        UNN_bd = self.DNN2Solu(XY_bd, scale=self.factor2freq, sFourier=self.sFourier)
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
        sum2WB = self.DNN2Solu.get_regular_sum2WB(regular_model=self.opt2regular_WB)
        return sum2WB

    def evalulate_MscaleDNN(self, XY_points=None):
        """
        Evaluating the MscaleDNN for testing points
        Args:
            XY_points: the testing input data of variable. ------  float, shape=[B,D]
        return:
            UNN: the testing output
        """
        assert (XY_points is not None)
        shape2XY = XY_points.shape
        lenght2XY_shape = len(shape2XY)
        assert (lenght2XY_shape == 2)
        assert (shape2XY[-1] == 2)

        UNN = self.DNN2Solu(XY_points, scale=self.factor2freq, sFourier=self.sFourier)
        return UNN


def solve_Multiscale_PDE(R):
    log_out_path = R['FolderName']         # 将路径从字典 R 中提取出来
    if not os.path.exists(log_out_path):   # 判断路径是否已经存在
        os.mkdir(log_out_path)             # 无 log_out_path 路径，创建一个 log_out_path 路径
    logfile_name = '%s_%s.txt' % ('log2train', R['name2act_hidden'])
    log_fileout = open(os.path.join(log_out_path, logfile_name), 'w')  # 在这个路径下创建并打开一个可写的 log_train.txt文件
    DNN_Log_Print.dictionary_out2file(R, log_fileout)

    DNN_tools.log_string('The penalty of flux(dual) variable": %s\n\n' % (R['gradient_penalty']), log_fileout)

    # 问题需要的设置
    batchsize_it = R['batch_size2interior']
    batchsize_bd = R['batch_size2boundary']

    bd_penalty_init = R['init_boundary_penalty']  # Regularization parameter for boundary conditions
    penalty2WB = R['penalty2weight_biases']       # Regularization parameter for weights and biases
    learning_rate = R['learning_rate']

    penalty2gd = R['gradient_penalty']

    input_dim = R['input_dim']
    out_dim = R['output_dim']

    # pLaplace 算子需要的额外设置, 先预设一下
    p_index = 2
    epsilon = 0.1
    mesh_number = 2
    region_lb = 0.0
    region_rt = 1.0

    if R['PDE_type'] == 'pLaplace_implicit':
        p_index = R['order2pLaplace_operator']
        epsilon = R['epsilon']
        mesh_number = R['mesh_number']
        if R['equa_name'] == 'multi_scale2D_5':
            region_lb = 0.0
            region_rt = 1.0
        else:
            region_lb = -1.0
            region_rt = 1.0
        u_true, f, A_eps, u_left, u_right, u_bottom, u_top = MS_LaplaceEqs.get_infos2pLaplace_2D(
            input_dim=input_dim, out_dim=out_dim, mesh_number=R['mesh_number'], pow_order2Aeps=R['order2Aeps_MSE4'],
            intervalL=-1.0, intervalR=1.0, equa_name=R['equa_name'])
    elif R['PDE_type'] == 'pLaplace_explicit':
        p_index = R['order2pLaplace_operator']
        epsilon = R['epsilon']
        mesh_number = R['mesh_number']
        if R['equa_name'] == 'multi_scale2D_7':
            region_lb = 0.0
            region_rt = 1.0
            u_true = MS_LaplaceEqs.true_solution2E7(input_dim, out_dim, eps=epsilon)
            u_left, u_right, u_bottom, u_top = MS_LaplaceEqs.boundary2E7(
                input_dim, out_dim, region_lb, region_rt, eps=epsilon)
            A_eps = MS_LaplaceEqs.elliptic_coef2E7(input_dim, out_dim, eps=epsilon)
    elif R['PDE_type'] == 'Possion_Boltzmann':
        # 求解如下方程, A_eps(x) 震荡的比较厉害，具有多个尺度
        #       d      ****         d         ****
        #   -  ----   |  A_eps(x)* ---- u_eps(x) | + Ku_eps =f(x), x \in R^n
        #       dx     ****         dx        ****
        # region_lb = -1.0
        region_lb = 0.0
        region_rt = 1.0
        p_index = R['order2pLaplace_operator']
        epsilon = R['epsilon']
        mesh_number = R['mesh_number']
        A_eps, kappa, u_true, u_left, u_right, u_top, u_bottom, f = MS_BoltzmannEqs.get_infos2Boltzmann_2D(
            equa_name=R['equa_name'], intervalL=region_lb, intervalR=region_rt)

    model = MscaleDNN(input_dim=R['input_dim'], out_dim=R['output_dim'], hidden_layer=R['hidden_layers'],
                      Model_name=R['model2NN'], name2actIn=R['name2act_in'], name2actHidden=R['name2act_hidden'],
                      name2actOut=R['name2act_out'], opt2regular_WB='L0', type2numeric='float32',
                      factor2freq=R['freq'], sFourier=R['sfourier'], repeat_highFreq=R['repeat_High_freq'],
                      use_gpu=R['use_gpu'], No2GPU=R['gpuNo'])
    if True == R['use_gpu']:
        model = model.cuda(device='cuda:'+str(R['gpuNo']))

    params2SoluDNN = model.DNN2Solu.parameters()
    params2PXDNN = model.DNN2PNNY.parameters()
    params2PYDNN = model.DNN2PNNY.parameters()

    params2Net = itertools.chain(params2SoluDNN, params2PXDNN, params2PYDNN)

    # 定义优化方法，并给定初始学习率
    # optimizer = torch.optim.SGD(params2Net, lr=init_lr)                     # SGD
    # optimizer = torch.optim.SGD(params2Net, lr=init_lr, momentum=0.8)       # momentum
    # optimizer = torch.optim.RMSprop(params2Net, lr=init_lr, alpha=0.95)     # RMSProp
    optimizer = torch.optim.Adam(params2Net, lr=learning_rate)                # Adam

    # 定义更新学习率的方法
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1/(epoch+1))
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 25, gamma=0.995)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 50, gamma=0.99)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 100, gamma=0.975)

    loss_it_all, loss_bd_all, loss_all, train_mse_all, train_rel_all = [], [], [], [], []  # 空列表, 使用 append() 添加元素
    test_mse_all, test_rel_all = [], []
    loss_adu_all = []
    test_epoch = []

    if R['testData_model'] == 'random_generate':
        # 生成测试数据，用于测试训练后的网络
        test_bach_size = 1600
        size2test = 40
        # test_bach_size = 4900
        # size2test = 70
        # test_bach_size = 10000
        # size2test = 100
        test_xy_bach = dataUtilizer2torch.rand_it(test_bach_size, R['input_dim'], region_lb, region_rt, to_torch=False,
                                                  to_float=True, to_cuda=False, gpu_no=R['gpuNo'])
        saveData.save_testData_or_solus2mat(test_xy_bach, dataName='testXY', outPath=R['FolderName'])
    else:
        if R['PDE_type'] == 'pLaplace_implicit' or R['PDE_type'] == 'pLaplace_explicit':
            test_xy_bach = Load_data2Mat.get_data2pLaplace(equation_name=R['equa_name'], mesh_number=mesh_number)
            size2batch = np.shape(test_xy_bach)[0]
            size2test = int(np.sqrt(size2batch))
            saveData.save_meshData2mat(test_xy_bach, dataName='meshXY', mesh_number=mesh_number, outPath=R['FolderName'])
        elif R['PDE_type'] == 'Possion_Boltzmann':
            if region_lb == (-1.0) and region_rt == 1.0:
                name2data_file = '11'
            else:
                name2data_file = '01'
            test_xy_bach = Load_data2Mat.get_meshData2Boltzmann(domain_lr=name2data_file, mesh_number=mesh_number)
            size2batch = np.shape(test_xy_bach)[0]
            size2test = int(np.sqrt(size2batch))
            saveData.save_meshData2mat(test_xy_bach, dataName='meshXY', mesh_number=mesh_number,
                                       outPath=R['FolderName'])
        elif R['PDE_type'] == 'Convection_diffusion':
            if region_lb == (-1.0) and region_rt == 1.0:
                name2data_file = '11'
            else:
                name2data_file = '01'
            test_xy_bach = Load_data2Mat.get_meshData2Boltzmann(domain_lr=name2data_file, mesh_number=mesh_number)
            size2batch = np.shape(test_xy_bach)[0]
            size2test = int(np.sqrt(size2batch))
            saveData.save_meshData2mat(test_xy_bach, dataName='meshXY', mesh_number=mesh_number, outPath=R['FolderName'])
        else:
            test_xy_bach = Load_data2Mat.get_randomData2mat(dim=R['input_dim'], data_path='dataMat_highDim')
            size2batch = np.shape(test_xy_bach)[0]
            size2test = int(np.sqrt(size2batch))

    test_xy_bach = test_xy_bach.astype(np.float32)
    test_xy_torch = torch.from_numpy(test_xy_bach)
    if True == R['use_gpu']:
        test_xy_torch = test_xy_torch.cuda(device='cuda:' + str(R['gpuNo']))

    t0 = time.time()
    for i_epoch in range(R['max_epoch'] + 1):
        # x_it_batch = dataUtilizer2torch.rand_it(batchsize_it, 1, region_a=region_lb, region_b=region_rt,
        #                                         to_float=True, to_cuda=R['use_gpu'], gpu_no=R['gpuNo'],
        #                                         use_grad2x=True)
        # y_it_batch = dataUtilizer2torch.rand_it(batchsize_it, 1, region_a=region_lb, region_b=region_rt,
        #                                         to_float=True, to_cuda=R['use_gpu'], gpu_no=R['gpuNo'],
        #                                         use_grad2x=True)
        #
        # xl_bd_batch, xr_bd_batch, yb_bd_batch, yt_bd_batch = dataUtilizer2torch.rand_bd_2D(
        #     batchsize_bd, R['input_dim'], region_a=region_lb, region_b=region_rt, to_float=True, to_cuda=R['use_gpu'],
        #     gpu_no=R['gpuNo'])
        x_it_batch = dataUtilizer2torch.rand_it_lhs(batchsize_it, 1, region_a=region_lb, region_b=region_rt,
                                                    to_float=True, to_cuda=R['use_gpu'], gpu_no=R['gpuNo'],
                                                    use_grad2x=True)
        y_it_batch = dataUtilizer2torch.rand_it_lhs(batchsize_it, 1, region_a=region_lb,
                                                    region_b=region_rt,
                                                    to_float=True, to_cuda=R['use_gpu'], gpu_no=R['gpuNo'],
                                                    use_grad2x=True)

        xl_bd_batch, xr_bd_batch, yb_bd_batch, yt_bd_batch = dataUtilizer2torch.rand_bd_2D_lhs(
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

        UNN2train, loss_it, loss_dAU = model.loss_in2pLaplace(
            X=x_it_batch, Y=y_it_batch, fside=f, aeps=A_eps, loss_type=R['loss_type'], scale2lncosh=R['scale2lncosh'])

        loss_bd2left = model.loss2bd(XY_bd=xl_bd_batch, Ubd_exact=u_left,
                                     loss_type=R['loss_type2bd'], scale2lncosh=R['scale2lncosh'])
        loss_bd2right = model.loss2bd(XY_bd=xr_bd_batch, Ubd_exact=u_right,
                                      loss_type=R['loss_type2bd'], scale2lncosh=R['scale2lncosh'])
        loss_bd2bottom = model.loss2bd(XY_bd=yb_bd_batch, Ubd_exact=u_bottom,
                                       loss_type=R['loss_type2bd'], scale2lncosh=R['scale2lncosh'])
        loss_bd2top = model.loss2bd(XY_bd=yt_bd_batch, Ubd_exact=u_top,
                                    loss_type=R['loss_type2bd'], scale2lncosh=R['scale2lncosh'])
        loss_bd = loss_bd2left + loss_bd2right + loss_bd2bottom + loss_bd2top

        PWB = penalty2WB * model.get_regularSum2WB()

        loss = loss_it + penalty2gd*loss_dAU + temp_penalty_bd * loss_bd + PWB  # 要优化的loss function

        loss_it_all.append(loss_it.item())
        loss_bd_all.append(loss_bd.item())
        loss_adu_all.append(loss_dAU.item())
        loss_all.append(loss.item())

        optimizer.zero_grad()               # 求导前先清零, 只要在下一次求导前清零即可
        loss.backward(retain_graph=True)    # 对loss关于Ws和Bs求偏导
        optimizer.step()                    # 更新参数Ws和Bs
        scheduler.step()

        if R['PDE_type'] == 'pLaplace_implicit':
            train_mse = torch.tensor([0], dtype=torch.float32)
            train_rel = torch.tensor([0], dtype=torch.float32)
        else:
            Uexact2train = u_true(torch.reshape(x_it_batch, shape=[-1, 1]),
                                  torch.reshape(y_it_batch, shape=[-1, 1]))
            train_mse = torch.mean(torch.square(UNN2train - Uexact2train))
            train_rel = train_mse / torch.mean(torch.square(Uexact2train))

        train_mse_all.append(train_mse.item())
        train_rel_all.append(train_rel.item())

        if i_epoch % 1000 == 0:
            run_times = time.time() - t0
            tmp_lr = optimizer.param_groups[0]['lr']
            DNN_tools.print_and_log_train_one_epoch(
                i_epoch, run_times, tmp_lr, temp_penalty_bd, PWB, loss_it.item(), loss_dAU.item(), loss_bd.item(),
                loss.item(), train_mse.item(), train_rel.item(), log_out=log_fileout)

            test_epoch.append(i_epoch / 1000)
            if R['PDE_type'] == 'pLaplace_implicit':
                UNN2test = model.evalulate_MscaleDNN(XY_points=test_xy_torch)
                Utrue2test = torch.from_numpy(u_true.astype(np.float32))
                Utrue2test = Utrue2test.cuda(device='cuda:' + str(R['gpuNo']))
            else:
                UNN2test = model.evalulate_MscaleDNN(XY_points=test_xy_torch)
                Utrue2test = u_true(torch.reshape(test_xy_torch[:, 0], shape=[-1, 1]),
                                    torch.reshape(test_xy_torch[:, 1], shape=[-1, 1]))

            point_square_error = torch.square(Utrue2test - UNN2test)
            test_mse = torch.mean(point_square_error)
            test_rel = test_mse / torch.mean(torch.square(Utrue2test))
            test_mse_all.append(test_mse.item())
            test_rel_all.append(test_rel.item())
            DNN_tools.print_and_log_test_one_epoch(test_mse.item(), test_rel.item(), log_out=log_fileout)

    # ------------------- save the training results into mat file and plot them -------------------------
    saveData.save_trainLoss2mat(loss_adu_all, lossName='Loss_Adu', actName=R['name2act_hidden'],
                                outPath=R['FolderName'])
    saveData.save_trainLoss2mat_1actFunc(loss_it_all, loss_bd_all, loss_all, actName=R['name2act_hidden'],
                                         outPath=R['FolderName'])
    saveData.save_train_MSE_REL2mat(train_mse_all, train_rel_all, actName=R['name2act_hidden'], outPath=R['FolderName'])

    plotData.plotTrain_loss_1act_func(loss_it_all, lossType='loss_it', seedNo=R['seed'], outPath=R['FolderName'],
                                      yaxis_scale=True)
    plotData.plotTrain_loss_1act_func(loss_bd_all, lossType='loss_bd', seedNo=R['seed'], outPath=R['FolderName'],
                                      yaxis_scale=True)
    plotData.plotTrain_loss_1act_func(loss_adu_all, lossType='loss_Adu', seedNo=R['seed'], outPath=R['FolderName'],
                                      yaxis_scale=True)
    plotData.plotTrain_loss_1act_func(loss_all, lossType='loss', seedNo=R['seed'], outPath=R['FolderName'],
                                      yaxis_scale=True)

    saveData.save_train_MSE_REL2mat(train_mse_all, train_rel_all, actName=R['name2act_hidden'], outPath=R['FolderName'])
    if True == R['use_gpu']:
        plotData.plotTrain_MSE_REL_1act_func(train_mse_all, train_rel_all, actName=R['name2act_hidden'],
                                             seedNo=R['seed'], outPath=R['FolderName'])
    else:
        plotData.plotTrain_MSE_REL_1act_func(train_mse_all, train_rel_all, actName=R['name2act_hidden'],
                                             seedNo=R['seed'], outPath=R['FolderName'], yaxis_scale=True)

    # ----------------------  save testing results to mat files, then plot them --------------------------------
    if True == R['use_gpu']:
        utrue2test_numpy = Utrue2test.cpu().detach().numpy()
        unn2test_numpy = UNN2test.cpu().detach().numpy()
        point_square_error_numpy = point_square_error.cpu().detach().numpy()
    else:
        utrue2test_numpy = Utrue2test.detach().numpy()
        unn2test_numpy = UNN2test.detach().numpy()
        point_square_error_numpy = point_square_error.detach().numpy()

    saveData.save_2testSolus2mat(utrue2test_numpy, unn2test_numpy, actName='utrue',
                                 actName1=R['name2act_hidden'], outPath=R['FolderName'])

    saveData.save_testMSE_REL2mat(test_mse_all, test_rel_all, actName=R['name2act_hidden'], outPath=R['FolderName'])
    plotData.plotTest_MSE_REL(test_mse_all, test_rel_all, test_epoch, actName=R['name2act_hidden'],
                              seedNo=R['seed'], outPath=R['FolderName'], yaxis_scale=True)

    saveData.save_test_point_wise_err2mat(point_square_error_numpy, actName=R['name2act_hidden'],
                                          outPath=R['FolderName'])

    plotData.plot_Hot_point_wise_err(point_square_error_numpy, size_vec2mat=size2test,
                                     actName=R['name2act_hidden'], seedNo=R['seed'], outPath=R['FolderName'])


if __name__ == "__main__":
    R={}
    R['gpuNo'] = 0
    if platform.system() == 'Windows':
        os.environ["CDUA_VISIBLE_DEVICES"] = "%s" % (R['gpuNo'])
    else:
        print('-------------------------------------- linux -----------------------------------------------')
        # Linux终端没有GUI, 需要添加如下代码，而且必须添加在 import matplotlib.pyplot 之前，否则无效。
        matplotlib.use('Agg')

    # 文件保存路径设置
    file2results = 'Results'
    store_file = 'FMPINN2D_X_Y'
    # store_file = 'Boltzmann2D'
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
    seed_str = str(R['seed'])  # int 型转为字符串型
    FolderName = os.path.join(OUT_DIR, seed_str)  # 路径连接
    R['FolderName'] = FolderName
    if not os.path.exists(FolderName):
        print('--------------------- FolderName -----------------:', FolderName)
        os.mkdir(FolderName)

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  复制并保存当前文件 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if platform.system() == 'Windows':
        shutil.copy(__file__, '%s/%s' % (FolderName, os.path.basename(__file__)))
    else:
        shutil.copy(__file__, '%s/%s' % (FolderName, os.path.basename(__file__)))

    # if the value of step_stop_flag is not 0, it will activate stop condition of step to kill program
    # step_stop_flag = input('please input an  integer number to activate step-stop----0:no---!0:yes--:')
    # R['activate_stop'] = int(step_stop_flag)
    R['activate_stop'] = int(0)
    # if the value of step_stop_flag is not 0, it will activate stop condition of step to kill program
    # R['max_epoch'] = 100000
    R['max_epoch'] = 50000
    # R['max_epoch'] = 60000
    if 0 != R['activate_stop']:
        epoch_stop = input('please input a stop epoch:')
        R['max_epoch'] = int(epoch_stop)

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Setup of multi-scale problem %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    R['input_dim'] = 2  # 输入维数，即问题的维数(几元问题)
    R['output_dim'] = 1  # 输出维数

    R['PDE_type'] = 'pLaplace_implicit'
    # R['equa_name'] = 'multi_scale2D_1'      # p=2 区域为 [-1,1]X[-1,1]
    # R['equa_name'] = 'multi_scale2D_2'      # p=2 区域为 [-1,1]X[-1,1]
    # R['equa_name'] = 'multi_scale2D_3'      # p=2 区域为 [-1,1]X[-1,1] 论文中的例子
    # R['equa_name'] = 'multi_scale2D_4'        # p=2 区域为 [-1,1]X[-1,1] 论文中的例子
    # R['equa_name'] = 'multi_scale2D_5'      # p=3 区域为 [0,1]X[0,1]   和例三的系数A一样
    # R['equa_name'] = 'multi_scale2D_6'      # p=3 区域为 [-1,1]X[-1,1] 和例三的系数A一样
    R['equa_name'] = 'multi_scale2D_9'  # p=2 区域为 [-1,1]X[-1,1] 论文中的例子
    # R['equa_name'] = 'multi_scale2D_10'  # p=2 区域为 [-1,1]X[-1,1] 论文中的例子

    # R['PDE_type'] = 'pLaplace_explicit'
    # R['equa_name'] = 'multi_scale2D_7'      # p=2 区域为 [0,1]X[0,1]

    R['mesh_number'] = 6
    R['epsilon'] = 0.1
    R['order2pLaplace_operator'] = 2

    R['order2Aeps_MSE4'] = 5

    if R['PDE_type'] == 'pLaplace_implicit':
        # 网格大小设置
        # mesh_number = input('please input mesh_number =')  # 由终端输入的会记录为字符串形式
        # R['mesh_number'] = int(mesh_number)                # 字符串转为浮点

        # R['mesh_number'] = 6
        R['mesh_number'] = 7
    elif R['PDE_type'] == 'Possion_Boltzmann' or R['PDE_type'] == 'pLaplace_explicit' \
            or R['PDE_type'] == 'Convection_diffusion':
        R['mesh_number'] = int(6)
        R['order2pLaplace_operator'] = float(2)

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Setup of DNN %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # 训练集的设置(内部和边界)
    if R['PDE_type'] == 'pLaplace_implicit':
        # R['batch_size2interior'] = 3000      # 内部训练数据的批大小
        R['batch_size2interior'] = 5000  # 内部训练数据的批大小
        # R['batch_size2interior'] = 7500  # 内部训练数据的批大小
        # R['batch_size2interior'] = 10000   # 内部训练数据的批大小
        if R['mesh_number'] == 2:
            R['batch_size2boundary'] = 25    # 边界训练数据的批大小
        elif R['mesh_number'] == 3:
            R['batch_size2boundary'] = 100   # 边界训练数据的批大小
        elif R['mesh_number'] == 4:
            R['batch_size2boundary'] = 200   # 边界训练数据的批大小
        elif R['mesh_number'] == 5:
            R['batch_size2boundary'] = 300   # 边界训练数据的批大小
        elif R['mesh_number'] == 6:
            R['batch_size2boundary'] = 500   # 边界训练数据的批大小
        elif R['mesh_number'] == 7:
            R['batch_size2boundary'] = 750   # 边界训练数据的批大小
    else:
        R['batch_size2interior'] = 5000      # 内部训练数据的批大小
        R['batch_size2boundary'] = 500       # 边界训练数据的批大小

    # 装载测试数据模式
    R['testData_model'] = 'loadData'
    # R['testData_model'] = 'random_generate'

    R['loss_type'] = 'L2_loss'                             # loss类型:L2 loss
    # R['loss_type'] = 'lncosh_loss'

    R['scale2lncosh'] = 0.01
    # R['scale2lncosh'] = 0.025
    # R['scale2lncosh'] = 0.05
    # R['scale2lncosh'] = 0.1
    # R['scale2lncosh'] = 0.5
    # R['scale2lncosh'] = 1

    R['loss_type2bd'] = 'l2_loss'

    if R['loss_type'] == 'lncosh_loss':
        R['loss_type2bd'] = 'lncosh_loss'

    R['optimizer_name'] = 'Adam'     # 优化器
    R['learning_rate'] = 0.01        # 学习率   0.01的学习率比0.005要好
    # R['learning_rate'] = 0.005     # 学习率
    R['learning_rate_decay'] = 5e-5  # 学习率 decay
    R['train_model'] = 'union_training'
    # R['train_model'] = 'group2_training'
    # R['train_model'] = 'group3_training'

    # 正则化权重和偏置的模式
    R['regular_wb_model'] = 'L0'
    # R['regular_wb_model'] = 'L1'
    # R['regular_wb_model'] = 'L2'
    R['penalty2weight_biases'] = 0.000                    # Regularization parameter for weights
    # R['penalty2weight_biases'] = 0.001                  # Regularization parameter for weights
    # R['penalty2weight_biases'] = 0.0025                 # Regularization parameter for weights

    # 边界的惩罚处理方式,以及边界的惩罚因子
    # R['activate_penalty2bd_increase'] = 0
    R['activate_penalty2bd_increase'] = 1
    
    # R['init_boundary_penalty'] = 1000                   # Regularization parameter for boundary conditions
    # R['init_boundary_penalty'] = 100                    # Regularization parameter for boundary conditions
    R['init_boundary_penalty'] = 10                       # Regularization parameter for boundary conditions

    # R['gradient_penalty'] = 5       # 这个因子最好                     # Regularization parameter for boundary conditions
    R['gradient_penalty'] = 10       # 这个因子最好                     # Regularization parameter for boundary conditions
    # R['gradient_penalty'] = 15      # 这个因子最好                     # Regularization parameter for boundary conditions
    # R['gradient_penalty'] = 20  # Regularization parameter for boundary conditions
    # R['gradient_penalty'] = 25  # Regularization parameter for boundary conditions

    # &&&&&&&&&&&&&&&&&&& 使用的网络模型 &&&&&&&&&&&&&&&&&&&&&&&&&&&
    # R['model2NN'] = 'Fourier_SubDNN'
    R['model2NN'] = 'Fourier_DNN'

    # &&&&&&&&&&&&&& 隐藏层的层数和每层神经元数目 网络的频率范围设置&&&&&&&&&&&&&&&&&&&&&&&
    if R['model2NN'] == 'Fourier_SubDNN':
        R['hidden_layers'] = (20, 30, 20, 20, 20)
        # R['hidden_layers'] = (50, 80, 60, 60, 30)
        # R['hidden_layers'] = (40, 60, 40, 40, 30)
        # R['hidden_layers'] = (40, 60, 40, 40, 40)
        # R['hidden_layers'] = (40, 80, 40, 40, 40)
        # R['hidden_layers'] = (50, 80, 50, 50, 40)
        # R['freq'] = np.concatenate(([1], np.arange(start=2, stop=51, step=2)), axis=0)
        # R['freq'] = np.concatenate(([1], np.arange(start=2, stop=65, step=3)), axis=0)
        # R['freq'] = np.concatenate(([1], np.arange(start=2, stop=111, step=5)), axis=0)
        # R['freq'] = np.concatenate(([1], np.arange(start=2, stop=100, step=4), [100]), axis=0)

        R['freq'] = np.concatenate(([1, 2, 3, 4, 5], np.arange(start=10, stop=100, step=5), [100]), axis=0)
    else:
        R['hidden_layers'] = (125, 150, 100, 80, 80)
        R['freq'] = np.concatenate(([1], np.arange(start=2, stop=101, step=2)), axis=0)

    # &&&&&&&&&&&&&&&&&&& 激活函数的选择 &&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    # R['name2act_in'] = 'relu'
    # R['name2act_in'] = 'tanh'
    # R['name2act_in'] = 'sin'
    R['name2act_in'] = 'sinAddcos'
    # R['name2act_in'] = 'enhance_tanh'
    # R['name2act_in'] = 's2relu'

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

    # R['sfourier'] = 0.5   # 0.5效果不如1.0
    R['sfourier'] = 1.0

    R['use_gpu'] = True

    R['repeat_High_freq'] = True

    solve_Multiscale_PDE(R)

