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

from Networks import DNN_base

from Problems import General_Laplace
from Problems import MS_LaplaceEqs

from utilizers import dataUtilizer2torch
from utilizers import DNN_tools
from utilizers import DNN_Log_Print
from utilizers import plotData
from utilizers import saveData


class FMPINN(tn.Module):
    def __init__(self, input_dim=4, out_dim=1, hidden_layer=None, Model_name='DNN', name2actIn='relu',
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

        self.mat2U = torch.tensor([[1], [0]], dtype=self.float_type, device=self.opt2device)  # 3 行 1 列
        self.mat2P = torch.tensor([[0], [1]], dtype=self.float_type, device=self.opt2device)  # 3 行 1 列

    def loss_in2pLaplace(self, X=None, fside=None, if_lambda2fside=True, aeps=None, if_lambda2aeps=True,
                         loss_type='ritz_loss', scale2lncosh=0.5):
        """
        Calculating the loss of Laplace equation with p=2 in the interior points for given domain
        -div(a·grad U) = f,   in Omega
        BU = g                on Partial Omega, where B is a boundary operator, g is a given function
        Args:
             XY:              the input data of variable. ------  float, shape=[B,D]
             fside:           the force side              ------  float, shape=[B,1]
             if_lambda2fside: the force-side is a lambda function or an array  ------ Bool
             aeps:            the multi-scale coefficient       -----  float, shape=[B,1]
             if_lambda2aeps:  the multi-scale coefficient is a lambda function or an array  ------ Bool
             loss_type:       the type of loss function(Ritz, L2, Lncosh)      ------ string
             scale2lncosh:    if the loss is lncosh, using it                  ------- float
        return:
             UNN:             the output data
             loss2func:       the output loss in the interior points for given domain
             loss2dAU:        the difference of divergence
        """
        assert (X is not None)
        assert (fside is not None)

        shape2X = X.shape
        lenght2X_shape = len(shape2X)
        assert (lenght2X_shape == 2)
        assert (shape2X[-1] == 1)

        if if_lambda2fside:
            force_side = fside(X)
        else:
            force_side = fside

        # obtaining the solution and calculating it gradient
        UPNN = self.DNN(X, scale=self.factor2freq, sFourier=self.sFourier)
        UNN = torch.matmul(UPNN, self.mat2U)
        PNN = torch.matmul(UPNN, self.mat2P)

        grad2UNNx = torch.autograd.grad(UNN, X, grad_outputs=torch.ones_like(X),
                                        create_graph=True, retain_graph=True)
        dUNN = torch.reshape(grad2UNNx[0], shape=[-1, 1])

        gradPNN = torch.autograd.grad(PNN, X, grad_outputs=torch.ones_like(X), create_graph=True,
                                      retain_graph=True)[0]

        dPNN = torch.reshape(gradPNN[0], shape=[-1, 1])

        if if_lambda2aeps:
            aeps_side = aeps(X)
        else:
            aeps_side = aeps
        loss2dAU_temp = PNN - torch.multiply(aeps_side, dUNN)

        loss2func_temp = -dPNN - force_side

        # calculating the loss
        try:
            if str.lower(loss_type) == 'l2_loss':
                square_loss2func = torch.mul(loss2func_temp, loss2func_temp)
                square_loss2dAU = torch.square(loss2dAU_temp)

                loss2func = torch.mean(square_loss2func)
                loss2dAU  = torch.mean(square_loss2dAU)
            elif str.lower(loss_type) == 'lncosh_loss':
                lncosh_loss2func = (1 / scale2lncosh) * torch.log(torch.cosh(scale2lncosh * loss2func_temp))
                lncosh_loss2dAU = (1 / scale2lncosh) * torch.log(torch.cosh(scale2lncosh * loss2dAU_temp))
                loss2func = torch.mean(lncosh_loss2func)
                loss2dAU = torch.mean(lncosh_loss2dAU)
            return UNN, loss2func, loss2dAU
        except ValueError:
            print('Error type for loss or no loss')
            return

    def loss2bd(self, X_bd=None, Ubd_exact=None, if_lambda2Ubd=True, loss_type='l2_loss', scale2lncosh=0.5):
        """
        Calculating the loss of PDEs on the boundary points for given boundary
        BU = g            on Partial Omega, where B is a boundary operator, g is a given function
        Args:
            X_bd:          the input data of variable. ------  float, shape=[B,D]
            Ubd_exact:     the exact function or array for boundary condition
            if_lambda2Ubd: the Ubd_exact is a lambda function or an array  ------ Bool
            loss_type:     the type of loss function(Ritz, L2, Lncosh)      ------ string
            scale2lncosh:  if the loss is lncosh, using it                  ------- float
        return:
            loss_bd: the output loss on the boundary points for given boundary
        """
        assert (X_bd is not None)
        assert (Ubd_exact is not None)

        shape2X = X_bd.shape
        lenght2X_shape = len(shape2X)
        assert (lenght2X_shape == 2)
        assert (shape2X[-1] == 1)

        if if_lambda2Ubd:
            Ubd = Ubd_exact(X_bd)
        else:
            Ubd = Ubd_exact

        UPNN = self.DNN(X_bd, scale=self.factor2freq, sFourier=self.sFourier)
        UNN_bd = torch.matmul(UPNN, self.mat2U)
        diff2bd = UNN_bd - Ubd
        try:
            if str.lower(loss_type) == 'l2_loss':
                loss_bd_square = torch.mul(diff2bd, diff2bd)
                loss_bd = torch.mean(loss_bd_square)
            elif str.lower(loss_type) == 'lncosh_loss':
                loss_bd_square = (1 / scale2lncosh) * torch.log(torch.cosh(scale2lncosh * diff2bd))
                loss_bd = torch.mean(loss_bd_square)
            return loss_bd
        except ValueError:
            print('Error type for loss or no loss')
            return

    def get_regularSum2WB(self):
        """
        Calculating the regularization sum of weights and biases
        """
        sum2WB = self.DNN.get_regular_sum2WB(regular_model=self.opt2regular_WB)
        return sum2WB

    def evaluate_FMPINN(self, X_points=None):
        """
        Evaluating the FMPINN for testing points
        Args:
            X_points: the testing input data of variable. ------  float, shape=[B,D]
        return:
            UNN: the testing output
        """
        assert (X_points is not None)
        shape2X = X_points.shape
        lenght2X_shape = len(shape2X)
        assert (lenght2X_shape == 2)
        assert (shape2X[-1] == 1)

        UPNN = self.DNN(X_points, scale=self.factor2freq, sFourier=self.sFourier)
        UNN = torch.matmul(UPNN, self.mat2U)
        return UNN


def solve_Multiscale_PDE(R):
    log_out_path = R['FolderName']        # 将路径从字典 R 中提取出来
    if not os.path.exists(log_out_path):  # 判断路径是否已经存在
        os.mkdir(log_out_path)            # 无 log_out_path 路径，创建一个 log_out_path 路径
    logfile_name = '%s_%s.txt' % ('log2train', R['name2act_hidden'])
    log_fileout = open(os.path.join(log_out_path, logfile_name), 'w')  # 在这个路径下创建并打开一个可写的 log_train.txt文件
    DNN_Log_Print.dictionary_out2file(R, log_fileout)

    DNN_tools.log_string('The penalty of flux(dual) variable": %s\n\n' % (R['gradient_penalty']), log_fileout)

    # 问题需要的设置
    batchsize_it = R['batch_size2interior']
    batchsize_bd = R['batch_size2boundary']

    bd_penalty_init = R['init_boundary_penalty']  # Regularization parameter for boundary conditions
    penalty2WB = R['penalty2weight_biases']       # Regularization parameter for weights and biases
    init_lr = R['learning_rate']

    penalty2gd = R['gradient_penalty']

    # ------- set the problem --------
    region_l = 0.0
    region_r = 1.0
    if R['equa_name'] == 'multi_scale':
        utrue, f, A_eps, uleft, uright = MS_LaplaceEqs.get_infos2pLaplace1D(
            in_dim=R['input_dim'], out_dim=R['output_dim'], intervalL=region_l, intervalR=region_r,
            index2p=2, eps=R['epsilon'])
    elif R['equa_name'] == '3scale2':
        utrue, f, A_eps, uleft, uright = MS_LaplaceEqs.get_infos2pLaplace1D_3scale(
            in_dim=R['input_dim'], out_dim=R['output_dim'], intervalL=region_l, intervalR=region_r,
            index2p=2, eps1=R['epsilon'], eps2=R['epsilon2'])

    model = FMPINN(input_dim=R['input_dim'], out_dim=R['output_dim'], hidden_layer=R['hidden_layers'],
                   Model_name=R['model2NN'], name2actIn=R['name2act_in'], name2actHidden=R['name2act_hidden'],
                   name2actOut=R['name2act_out'], opt2regular_WB='L0', type2numeric='float32',
                   factor2freq=R['freq'], sFourier=R['sfourier'], repeat_highFreq=R['repeat_High_freq'],
                   use_gpu=R['use_gpu'], No2GPU=R['gpuNo'])
    if True == R['use_gpu']:
        model = model.cuda(device='cuda:'+str(R['gpuNo']))

    params2Net = model.DNN.parameters()

    # 定义优化方法，并给定初始学习率
    # optimizer = torch.optim.SGD(params2Net, lr=init_lr)                     # SGD
    # optimizer = torch.optim.SGD(params2Net, lr=init_lr, momentum=0.8)       # momentum
    # optimizer = torch.optim.RMSprop(params2Net, lr=init_lr, alpha=0.95)     # RMSProp
    optimizer = torch.optim.Adam(params2Net, lr=init_lr)  # Adam

    # 定义更新学习率的方法
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1/(epoch+1))
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 25, gamma=0.995)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 50, gamma=0.99)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 100, gamma=0.975)

    t0 = time.time()
    loss_it_all, loss_bd_all, loss_all, train_mse_all, train_rel_all = [], [], [], [], []  # 空列表, 使用 append() 添加元素
    test_mse_all, test_rel_all = [], []
    loss_dadu_all = []
    test_epoch = []
    test_batch_size = 5000
    test_x_bach = np.reshape(np.linspace(region_l, region_r, num=test_batch_size), [-1, 1])
    test_x_bach = test_x_bach.astype(np.float32)
    test_x_bach = torch.from_numpy(test_x_bach)
    if True == R['use_gpu']:
        test_x_bach = test_x_bach.cuda(device='cuda:' + str(R['gpuNo']))

    for i_epoch in range(R['max_epoch'] + 1):
        x_it_batch = dataUtilizer2torch.rand_in_1D(
            batch_size=batchsize_it, variable_dim=R['input_dim'], region_a=region_l, region_b=region_r,
            to_torch=True, to_float=True, to_cuda=R['use_gpu'], gpu_no=R['gpuNo'], use_grad=True)
        xl_bd_batch, xr_bd_batch = dataUtilizer2torch.rand_bd_1D(
            batch_size=batchsize_bd, variable_dim=R['input_dim'], region_a=region_l, region_b=region_r, to_torch=True,
            to_float=True, to_cuda=R['use_gpu'], gpu_no=R['gpuNo'])

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

        if R['equa_name'] == 'multi_scale':
            UNN2train, loss_it, loss_dAU = model.loss_in2pLaplace(
                X=x_it_batch, fside=f, if_lambda2fside=True, aeps=A_eps, loss_type=R['loss_type'],
                scale2lncosh=R['scale2lncosh'])
        else:
            f_3scale = MS_LaplaceEqs.force_side_3scale(x_it_batch, eps1=R['epsilon'], eps2=R['epsilon2'])
            UNN2train, loss_it, loss_dAU = model.loss_in2pLaplace(
                X=x_it_batch, fside=f_3scale, if_lambda2fside=False, aeps=A_eps, loss_type=R['loss_type'],
                scale2lncosh=R['scale2lncosh'])

        loss_bd2left = model.loss2bd(X_bd=xl_bd_batch, Ubd_exact=uleft, if_lambda2Ubd=True,
                                     loss_type=R['loss_type2bd'], scale2lncosh=R['scale2lncosh'])
        loss_bd2right = model.loss2bd(X_bd=xr_bd_batch, Ubd_exact=uright, if_lambda2Ubd=True,
                                      loss_type=R['loss_type2bd'], scale2lncosh=R['scale2lncosh'])
        loss_bd = loss_bd2left + loss_bd2right
        pwb = penalty2WB*model.get_regularSum2WB()

        loss = loss_it + penalty2gd*loss_dAU + temp_penalty_bd*loss_bd + pwb

        loss_all.append(loss.item())
        loss_it_all.append(loss_it.item())
        loss_bd_all.append(loss_bd.item())
        loss_dadu_all.append(loss_dAU.item())

        optimizer.zero_grad()  # 求导前先清零, 只要在下一次求导前清零即可
        loss.backward()        # 对loss关于Ws和Bs求偏导
        optimizer.step()       # 更新参数Ws和Bs
        scheduler.step()

        Uexact = utrue(x_it_batch)
        train_mse = torch.mean(torch.mul(UNN2train-Uexact, UNN2train-Uexact))
        train_rel = train_mse/torch.mean(torch.mul(Uexact, Uexact))

        train_mse_all.append(train_mse.item())
        train_rel_all.append(train_rel.item())

        if i_epoch % 1000 == 0:
            run_times = time.time() - t0
            tmp_lr = optimizer.param_groups[0]['lr']
            DNN_tools.print_and_log_train_one_epoch(
                i_epoch, run_times, tmp_lr, temp_penalty_bd, pwb, loss_it.item(), loss_dAU.item(), loss_bd.item(),
                loss.item(), train_mse.item(), train_rel.item(), log_out=log_fileout)

            test_epoch.append(i_epoch / 1000)
            unn2test = model.evaluate_FMPINN(X_points=test_x_bach)
            Uexact2test = utrue(test_x_bach)

            point_square_error = torch.square(Uexact2test - unn2test)
            test_mse = torch.mean(point_square_error)
            test_rel = test_mse/torch.mean(torch.mul(Uexact2test, Uexact2test))

            test_mse_all.append(test_mse.item())
            test_rel_all.append(test_rel.item())

            DNN_tools.print_and_log_test_one_epoch(test_mse.item(), test_rel.item(), log_out=log_fileout)

    # -----------------------  save training results to mat files, then plot them ---------------------------------
    saveData.save_trainLoss2mat(loss_dadu_all, lossName='Loss_Adu', actName=R['name2act_hidden'], outPath=R['FolderName'])
    saveData.save_trainLoss2mat_1actFunc(loss_it_all, loss_bd_all, loss_all, actName=R['name2act_hidden'],
                                         outPath=R['FolderName'])
    plotData.plotTrain_loss_1act_func(loss_it_all, lossType='loss_it', seedNo=R['seed'], outPath=R['FolderName'],
                                      yaxis_scale=True)
    plotData.plotTrain_loss_1act_func(loss_bd_all, lossType='loss_bd', seedNo=R['seed'], outPath=R['FolderName'],
                                      yaxis_scale=True)
    plotData.plotTrain_loss_1act_func(loss_dadu_all, lossType='loss_Adu', seedNo=R['seed'], outPath=R['FolderName'],
                                      yaxis_scale=True)
    plotData.plotTrain_loss_1act_func(loss_all, lossType='loss', seedNo=R['seed'], outPath=R['FolderName'],
                                      yaxis_scale=True)

    saveData.save_train_MSE_REL2mat(train_mse_all, train_rel_all, actName=R['name2act_hidden'], outPath=R['FolderName'])
    plotData.plotTrain_MSE_REL_1act_func(train_mse_all, train_rel_all, actName=R['name2act_hidden'], seedNo=R['seed'],
                                         outPath=R['FolderName'], yaxis_scale=True)

    # ----------------------  save testing results to mat files, then plot them --------------------------------
    if True == R['use_gpu']:
        numpy_Uexact2test = Uexact2test.cpu().detach().numpy()
        numpy_Unn2test = unn2test.cpu().detach().numpy()
        numpy_test_x_bach = test_x_bach.cpu().detach().numpy()
    else:
        numpy_Uexact2test = Uexact2test.detach().numpy()
        numpy_Unn2test = unn2test.detach().numpy()
        numpy_test_x_bach = test_x_bach.detach().numpy()

    saveData.save_2testSolus2mat(numpy_Uexact2test, numpy_Unn2test, actName='utrue',
                                 actName1=R['name2act_hidden'], outPath=R['FolderName'])
    plotData.plot_2solutions2test(numpy_Uexact2test, numpy_Unn2test, coord_points2test=numpy_test_x_bach,
                                  batch_size2test=test_batch_size, seedNo=R['seed'], outPath=R['FolderName'],
                                  subfig_type=R['subfig_type'])

    saveData.save_testMSE_REL2mat(test_mse_all, test_rel_all, actName=R['name2act_hidden'], outPath=R['FolderName'])
    plotData.plotTest_MSE_REL(test_mse_all, test_rel_all, test_epoch, actName=R['name2act_hidden'],
                              seedNo=R['seed'], outPath=R['FolderName'], yaxis_scale=True)


if __name__ == "__main__":
    R={}
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

    # ------------------------------------------- 文件保存路径设置 ----------------------------------------
    file2results = 'Results'
    store_file = 'FMPINN1D'
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

    # ---------------------------- Setup of laplace equation ------------------------------
    # if the value of step_stop_flag is not 0, it will activate stop condition of step to kill program
    # step_stop_flag = input('please input an  integer number to activate step-stop----0:no---!0:yes--:')
    # R['activate_stop'] = int(step_stop_flag)
    R['activate_stop'] = int(0)
    # if the value of step_stop_flag is not 0, it will activate stop condition of step to kill program
    R['max_epoch'] = 50000
    # R['max_epoch'] = 100000
    if 0 != R['activate_stop']:
        epoch_stop = input('please input a stop epoch:')
        R['max_epoch'] = int(epoch_stop)

    R['PDE_type'] = 'pLaplace'
    # R['equa_name'] = 'multi_scale'
    R['equa_name'] = '3scale2'
    # R['equa_name'] = 'rand_ceof'
    # R['equa_name'] = 'rand_sin_ceof'

    R['order2pLaplace_operator'] = 2

    if R['equa_name'] == 'multi_scale':
        epsilon = input('please input epsilon =')  # 由终端输入的会记录为字符串形式
        R['epsilon'] = float(epsilon)  # 字符串转为浮点

    if R['equa_name'] == '3scale2':
        R['epsilon'] = 0.1
        R['epsilon2'] = 0.01

    R['input_dim'] = 1                               # 输入维数，即问题的维数(几元问题)
    R['output_dim'] = 1+1                              # 输出维数

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Setup of DNN %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # 训练集的设置(内部和边界)
    R['batch_size2interior'] = 3000                  # 内部训练数据的批大小
    R['batch_size2boundary'] = 500                   # 边界训练数据大小

    # 装载测试数据模式和画图
    R['plot_ongoing'] = 0
    R['subfig_type'] = 1
    R['testData_model'] = 'loadData'
    # R['testData_model'] = 'random_generate'

    R['loss_type'] = 'L2_loss'                     # L2 loss
    # R['loss_type'] = 'lncosh_loss'                 # lncosh_loss

    # R['scale2lncosh'] = 0.01
    R['scale2lncosh'] = 0.025
    # R['scale2lncosh'] = 0.05
    # R['scale2lncosh'] = 0.1
    # R['scale2lncosh'] = 0.5
    # R['scale2lncosh'] = 1

    if R['loss_type'] == 'L2_loss':
        R['loss_type2bd'] = 'L2_loss'
        # R['batch_size2interior'] = 15000             # 内部训练数据的批大小
        # R['batch_size2boundary'] = 2500              # 边界训练数据大小
    if R['loss_type'] == 'lncosh_loss':
        # R['batch_size2interior'] = 15000             # 内部训练数据的批大小
        # R['batch_size2boundary'] = 2500              # 边界训练数据大小
        R['loss_type2bd'] = 'lncosh_loss'            # lncosh_loss

    R['optimizer_name'] = 'Adam'                     # 优化器
    # R['learning_rate'] = 2e-4                        # 学习率
    R['learning_rate'] = 0.01  # 学习率
    # R['learning_rate'] = 0.005  # 学习率

    R['learning_rate_decay'] = 5e-5                  # 学习率 decay
    R['train_model'] = 'union_training'

    R['regular_wb_model'] = 'L0'
    # R['regular_wb_model'] = 'L1'
    # R['regular_wb_model'] = 'L2'
    R['penalty2weight_biases'] = 0.000               # Regularization parameter for weights
    # R['penalty2weight_biases'] = 0.001             # Regularization parameter for weights
    # R['penalty2weight_biases'] = 0.0025            # Regularization parameter for weights

    # 边界的惩罚处理方式,以及边界的惩罚因子
    R['activate_penalty2bd_increase'] = 1
    # R['init_boundary_penalty'] = 1000              # Regularization parameter for boundary conditions
    # R['init_boundary_penalty'] = 100                 # Regularization parameter for boundary conditions
    R['init_boundary_penalty'] = 10  # Regularization parameter for boundary conditions

    # R['gradient_penalty'] = 1  # Regularization parameter for boundary conditions
    # R['gradient_penalty'] = 5  # Regularization parameter for boundary conditions
    R['gradient_penalty'] = 10  # 最初的设置                      # Regularization parameter for boundary conditions
    # R['gradient_penalty'] = 15  # Regularization parameter for boundary conditions
    # R['gradient_penalty'] = 20  # Regularization parameter for boundary conditions
    # R['gradient_penalty'] = 25  # Regularization parameter for boundary conditions

    # 此处分别设为1，5，10，15，20， 25。验证超参的影响，之前的实验已经做了10.接下来进行1，5，15，20，25的实验

    # 网络的频率范围设置
    R['freq'] = np.arange(1, 121)

    # &&&&&&&&&&&&&&&&&&& 使用的网络模型 &&&&&&&&&&&&&&&&&&&&&&&&&&&
    # R['model2NN'] = 'DNN'
    # R['model2NN'] = 'Scale_DNN'
    # R['model2NN'] = 'Adapt_scale_DNN'
    # R['model2NN'] = 'Fourier_DNN'
    R['model2NN'] = 'Fourier_SubDNN'

    # &&&&&&&&&&&&&&&&&&&&&& 隐藏层的层数和每层神经元数目 &&&&&&&&&&&&&&&&&&&&&&&&&&&&
    if R['model2NN'] == 'Fourier_DNN':
        R['hidden_layers'] = (125, 100, 80, 80, 60)  # 1*125+250*100+100*80+80*80+80*60+60*1= 44385 个参数
    elif R['model2NN'] == 'Fourier_SubDNN':
        # R['hidden_layers'] = (5, 10, 5, 5, 5)
        # R['hidden_layers'] = (10, 20, 10, 10, 10)
        # R['hidden_layers'] = (60, 20, 10, 10, 10)
        # R['hidden_layers'] = (20, 30, 20, 20, 20)
        # R['hidden_layers'] = (20, 40, 20, 20, 20)
        R['hidden_layers'] = (30, 40, 30, 30, 30)
        # R['hidden_layers'] = (30, 50, 30, 30, 30)
        # R['hidden_layers'] = (40, 60, 40, 40, 40)
        # R['hidden_layers'] = (20, 60, 40, 40, 30)
        # R['freq'] = np.concatenate(([1], np.arange(start=2, stop=100, step=5), [100]), axis=0)
        R['freq'] = np.concatenate(([1, 2, 3, 4, 5], np.arange(start=10, stop=100, step=5), [100]), axis=0)
    else:
        R['hidden_layers'] = (250, 100, 80, 80, 60)  # 1*250+250*100+100*80+80*80+80*60+60*1= 44510 个参数

    # &&&&&&&&&&&&&&&&&&& 激活函数的选择 &&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    # R['name2act_in'] = 'tanh'
    # R['name2act_in'] = 'sin'
    R['name2act_in'] = 'sinADDcos'
    # R['name2act_in'] = 'Enhance_tanh'
    # R['name2act_in'] = 'gelu'
    # R['name2act_in'] = 'ReQU'

    # R['name2act_hidden'] = 'relu'
    # R['name2act_hidden'] = 'tanh'
    # R['name2act_hidden'] = 'Enhance_tanh'
    # R['name2act_hidden'] = 'sin'
    # R['name2act_hidden'] = 'srelu'
    # R['name2act_hidden'] = 's2relu'
    R['name2act_hidden'] = 'sinADDcos'
    # R['name2act_hidden'] = 'ReQU'
    # R['name2act_hidden'] = 'elu'
    # R['name2act_hidden'] = 'gelu'
    # R['name2act_hidden'] = 'mgelu'
    # R['name2act_hidden'] = 'phi'

    R['name2act_out'] = 'linear'

    if R['model2NN'] == 'Fourier_DNN' and R['name2act_hidden'] == 'tanh':
        # R['sfourier'] = 0.5
        R['sfourier'] = 1.0
    elif R['model2NN'] == 'Fourier_DNN' and R['name2act_hidden'] == 's2relu':
        R['sfourier'] = 0.5
        # R['sfourier'] = 1.0
    elif R['model2NN'] == 'Fourier_DNN' and R['name2act_hidden'] == 'sinAddcos':
        # R['sfourier'] = 0.5
        R['sfourier'] = 1.0
    elif R['model2NN'] == 'Fourier_DNN' and R['name2act_hidden'] == 'sin':
        # R['sfourier'] = 0.5
        R['sfourier'] = 1.0
    else:
        R['sfourier'] = 1.0
        # R['sfourier'] = 5.0
        # R['sfourier'] = 0.75

    R['use_gpu'] = True
    R['repeat_High_freq'] = True
    solve_Multiscale_PDE(R)
