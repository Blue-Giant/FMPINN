"""
@author: LXA
 Date: 2020 年 5 月 31 日
"""
from utilizers import DNN_tools
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import ConnectionPatch
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LogNorm


# 对同一个网络所得到的单个loss数据画图。如只画 loss to boundary (loss_bd)
def plotTrain_loss_1act_func(data2loss, lossType=None, seedNo=1000, outPath=None, xaxis_scale=False, yaxis_scale=False):
    plt.figure()
    ax = plt.gca()
    plt.plot(data2loss, 'b-.', label=lossType)
    plt.xlabel('epoch', fontsize=14)
    plt.ylabel(lossType, fontsize=14)
    plt.legend(fontsize=18)
    if xaxis_scale:
        ax.set_yscale('log')
    if yaxis_scale:
        ax.set_yscale('log')
    # plt.title('loss_it', fontsize=15)
    fntmp = '%s/%s%s' % (outPath, seedNo, lossType)
    DNN_tools.mySaveFig(plt, fntmp, ax=ax, isax=1, iseps=0)


# 对同一个网络所得到的多个loss数据画图。如画 loss to boundary and loss to interior (loss_bd loss_it)
def plot_2Trainlosses_1act_func(data2loss_1, data2loss_2, lossName1=None, lossName2=None, seedNo=1000, outPath=None,
                                lossType=None, xaxis_scale=False, yaxis_scale=False):
    plt.figure()
    ax = plt.gca()
    plt.plot(data2loss_1, 'b-.', label=lossName1)
    plt.plot(data2loss_2, 'r:', label=lossName2)
    plt.xlabel('epoch', fontsize=14)
    plt.ylabel('loss', fontsize=14)
    plt.legend(fontsize=13)
    if xaxis_scale:
        ax.set_yscale('log')
    if yaxis_scale:
        ax.set_yscale('log')
    # plt.title('loss_it', fontsize=15)
    fntmp = '%s/%s%s' % (outPath, seedNo, lossType)
    DNN_tools.mySaveFig(plt, fntmp, ax=ax, isax=1, iseps=0)


# 对同一个网络所得到的多个loss数据画图。如画 loss to boundary and loss to interior
def plot_3Trainlosses_1act_func(data2loss_1, data2loss_2, data2loss_3, lossName1=None, lossName2=None, lossName3=None,
                                lossType=None, seedNo=1000, outPath=None, xaxis_scale=False, yaxis_scale=False):
    plt.figure()
    ax = plt.gca()
    plt.plot(data2loss_1, 'b-.', label=lossName1)
    plt.plot(data2loss_2, 'r:', label=lossName2)
    plt.plot(data2loss_3, 'c*', label=lossName3)
    plt.xlabel('epoch', fontsize=14)
    plt.ylabel('loss', fontsize=14)
    plt.legend(fontsize=13)
    if xaxis_scale:
        ax.set_yscale('log')
    if yaxis_scale:
        ax.set_yscale('log')
    # plt.title('loss_it', fontsize=15)
    fntmp = '%s/%s%s' % (outPath, seedNo, lossType)
    DNN_tools.mySaveFig(plt, fntmp, ax=ax, isax=1, iseps=0)


# 对同两个网络所得到的同一种类型的loss数据画图。如画 loss to boundary
def plotTrain_losses_2act_funs(data2loss_1, data2loss_2, lossName1=None, lossName2=None, lossType=None,
                               seedNo=1000, outPath=None, xaxis_scale=False, yaxis_scale=False):
    plt.figure()
    ax = plt.gca()
    plt.plot(data2loss_1, 'b-.', label=lossName1)
    plt.plot(data2loss_2, 'r:', label=lossName2)
    plt.xlabel('epoch', fontsize=14)
    plt.ylabel('loss', fontsize=14)
    plt.legend(fontsize=13)
    if xaxis_scale:
        ax.set_yscale('log')
    if yaxis_scale:
        ax.set_yscale('log')
    # plt.title('loss_it', fontsize=15)
    fntmp = '%s/%s%s' % (outPath, seedNo, lossType)
    DNN_tools.mySaveFig(plt, fntmp, ax=ax, isax=1, iseps=0)


# 对三个网络所得到的同一个类型的loss数据画图。如画 loss to boundary
def plotTrain_losses_2Type2(data2loss_1, data2loss_2, data2loss_3, lossName1=None, lossName2=None, lossName3=None,
                            lossType=None, seedNo=1000, outPath=None, xaxis_scale=False, yaxis_scale=False):
    plt.figure()
    ax = plt.gca()
    plt.plot(data2loss_1, 'b-.', label=lossName1)
    plt.plot(data2loss_2, 'r:', label=lossName2)
    plt.plot(data2loss_3, 'c*', label=lossName3)
    plt.xlabel('epoch', fontsize=14)
    plt.ylabel('loss', fontsize=14)
    plt.legend(fontsize=13)
    if xaxis_scale:
        ax.set_yscale('log')
    if yaxis_scale:
        ax.set_yscale('log')
    # plt.title('loss_it', fontsize=15)
    fntmp = '%s/%s%s' % (outPath, seedNo, lossType)
    DNN_tools.mySaveFig(plt, fntmp, ax=ax, isax=1, iseps=0)


# 这个函数可以由 plot_3losses_1Type(......)代替
def plotTrain_losses(loss2s2ReLU, loss2sReLU, loss2ReLU, lossType=None, seedNo=1000, outPath=None):
    if 'loss_it' == lossType:
        plt.figure()
        ax = plt.gca()
        plt.plot(loss2s2ReLU, 'b-.', label='s2ReLU')
        plt.plot(loss2sReLU, 'r:', label='sReLU')
        plt.plot(loss2ReLU, 'c-*', label='ReLU')
        plt.xlabel('epoch', fontsize=14)
        plt.ylabel('loss_it', fontsize=14)
        plt.legend(fontsize=13)
        # plt.title('loss_it', fontsize=15)
        fntmp = '%s/%sloss_it' % (outPath, seedNo)
        DNN_tools.mySaveFig(plt, fntmp, ax=ax, isax=1, iseps=0)
    elif 'loss_bd' == lossType:
        plt.figure()
        ax = plt.gca()
        plt.plot(loss2s2ReLU, 'b-.', label='s2ReLU')
        plt.plot(loss2sReLU, 'r:', label='sReLU')
        plt.plot(loss2ReLU, 'c-*', label='ReLU')
        ax.set_yscale('log')
        plt.xlabel('epoch', fontsize=14)
        plt.ylabel('loss_bd', fontsize=14)
        plt.legend(fontsize=13)
        # plt.title('loss_bd', fontsize=15)
        fntmp = '%s/%sloss_bd' % (outPath, seedNo)
        DNN_tools.mySaveFig(plt, fntmp, ax=ax, isax=1, iseps=0)
    elif 'loss' == lossType:
        plt.figure()
        ax = plt.gca()
        plt.plot(loss2s2ReLU, 'b-.', label='s2ReLU')
        plt.plot(loss2sReLU, 'r:', label='sReLU')
        plt.plot(loss2ReLU, 'c-*', label='ReLU')
        plt.xlabel('epoch', fontsize=14)
        plt.ylabel('loss', fontsize=14)
        plt.legend(fontsize=13)
        # plt.title('loss', fontsize=15)
        fntmp = '%s/%sloss' % (outPath, seedNo)
        DNN_tools.mySaveFig(plt, fntmp, ax=ax, isax=1, iseps=0)


def plotTrain_MSE_1act_func(data2mse, mseType=None,seedNo=1000, outPath=None, xaxis_scale=False, yaxis_scale=False):
    plt.figure()
    ax = plt.gca()
    plt.plot(data2mse, 'b-.', label=mseType)
    plt.xlabel('epoch', fontsize=14)
    plt.ylabel('error', fontsize=14)
    plt.legend(fontsize=13)
    if xaxis_scale:
        ax.set_yscale('log')
    if yaxis_scale:
        ax.set_yscale('log')
    # plt.title('training mse', fontsize=13)
    fntmp = '%s/%strain_mse' % (outPath, seedNo)
    DNN_tools.mySaveFig(plt, fntmp, ax=ax, isax=1, iseps=0)


def plotTrain_REL_1act_func(data2rel, relType=None, seedNo=1000, outPath=None, xaxis_scale=False, yaxis_scale=False):
    plt.figure()
    ax = plt.gca()
    plt.plot(data2rel, 'b-.', label=relType)
    plt.xlabel('epoch', fontsize=14)
    plt.ylabel('error', fontsize=14)
    plt.legend(fontsize=13)
    if xaxis_scale:
        ax.set_yscale('log')
    if yaxis_scale:
        ax.set_yscale('log')
    # plt.title('training mse', fontsize=13)
    fntmp = '%s/%strain_mse' % (outPath, seedNo)
    DNN_tools.mySaveFig(plt, fntmp, ax=ax, isax=1, iseps=0)


def plotTrain_MSE_REL_1act_func(data2mse, data2rel, actName=None, seedNo=1000, outPath=None, xaxis_scale=False,
                                yaxis_scale=False):
    plt.figure()
    ax = plt.gca()
    plt.plot(data2mse, 'r-.', label='MSE')
    plt.plot(data2rel, 'b:', label='REL')
    if xaxis_scale:
        ax.set_yscale('log')
    if yaxis_scale:
        ax.set_yscale('log')
    plt.xlabel('epoch', fontsize=18)
    plt.ylabel('error', fontsize=18)
    plt.legend(fontsize=18)
    # plt.title('training error', fontsize=15)
    if str.lower(actName) == 'srelu':
        fntmp = '%s/%strainErr_%s' % (outPath, seedNo, 'sReLU')
    elif str.lower(actName) == 'sin':
        fntmp = '%s/%strainErr_%s' % (outPath, seedNo, 'sin')
    elif str.lower(actName) == 's2relu':
        fntmp = '%s/%strainErr_%s' % (outPath, seedNo, 's2ReLU')
    elif str.lower(actName) == 's3relu':
        fntmp = '%s/%strainErr_%s' % (outPath, seedNo, 's3ReLU')
    elif str.lower(actName) == 'csrelu':
        fntmp = '%s/%strainErr_%s' % (outPath, seedNo, 'CsReLU')
    elif str.lower(actName) == 'relu':
        fntmp = '%s/%strainErr_%s' % (outPath, seedNo, 'ReLU')
    elif str.lower(actName) == 'elu':
        fntmp = '%s/%strainErr_%s' % (outPath, seedNo, 'elu')
    elif str.lower(actName) == 'tanh':
        fntmp = '%s/%strainErr_%s' % (outPath, seedNo, 'tanh')
    elif str.lower(actName) == 'sintanh':
        fntmp = '%s/%strainErr_%s' % (outPath, seedNo, 'sintanh')
    elif str.lower(actName) == 'singauss':
        fntmp = '%s/%strainErr_%s' % (outPath, seedNo, 'singauss')
    elif str.lower(actName) == 'gauss':
        fntmp = '%s/%strainErr_%s' % (outPath, seedNo, 'gauss')
    elif str.lower(actName) == 'mexican':
        fntmp = '%s/%strainErr_%s' % (outPath, seedNo, 'mexican')
    else:
        fntmp = '%s/trainErr_%s' % (outPath, str.lower(actName))
    DNN_tools.mySaveFig(plt, fntmp, ax=ax, isax=1, iseps=0)


def plotTrain_MSEs_2act_funcs(data2mse1, data2mse2, mseName1=None, mseName2=None, seedNo=1000, outPath=None,
                              xaxis_scale=False, yaxis_scale=False):
    plt.figure()
    ax = plt.gca()
    plt.plot(data2mse1, 'b-.', label=mseName1)
    plt.plot(data2mse2, 'r:', label=mseName2)
    plt.xlabel('epoch', fontsize=14)
    plt.ylabel('error', fontsize=14)
    plt.legend(fontsize=13)
    if xaxis_scale:
        ax.set_yscale('log')
    if yaxis_scale:
        ax.set_yscale('log')
    # plt.title('training mse', fontsize=13)
    fntmp = '%s/%strain_mses' % (outPath, seedNo)
    DNN_tools.mySaveFig(plt, fntmp, ax=ax, isax=1, iseps=0)


def plotTrain_RELs_2act_funcs(data2rel1, data2rel2, relName1=None, relName2=None, seedNo=1000, outPath=None,
                              xaxis_scale=False, yaxis_scale=False):
    plt.figure()
    ax = plt.gca()
    plt.plot(data2rel1, 'b-.', label=relName1)
    plt.plot(data2rel2, 'r:', label=relName2)
    plt.xlabel('epoch', fontsize=14)
    plt.ylabel('error', fontsize=14)
    plt.legend(fontsize=13)
    if xaxis_scale:
        ax.set_yscale('log')
    if yaxis_scale:
        ax.set_yscale('log')
    # plt.title('training mse', fontsize=13)
    fntmp = '%s/%strain_rels' % (outPath, seedNo)
    DNN_tools.mySaveFig(plt, fntmp, ax=ax, isax=1, iseps=0)


def plotTrain_MSEs_RELs_2act_funcs(mse2data1, mse2data2, rel2data1, rel2data2, actName1=None, actName2=None,
                                   seedNo=1000, outPath=None, xaxis_scale=False, yaxis_scale=False):
    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    plt.plot(mse2data1, 'g-.', label=str('MSE-'+actName1))
    plt.plot(rel2data1, 'b:', label=str('REL-'+actName1))
    plt.plot(mse2data2, 'm--.', label=str('MSE-'+actName2))
    plt.plot(rel2data2, 'c-*', label=str('REL-'+actName2))
    if xaxis_scale:
        ax.set_yscale('log')
    if yaxis_scale:
        ax.set_yscale('log')
    plt.xlabel('epoch', fontsize=14)
    plt.ylabel('error', fontsize=14)
    ax.legend(loc='center', bbox_to_anchor=(0.485, 1.055), ncol=3, fontsize=12)
    # plt.legend(fontsize=11)
    # plt.title(' train error', fontsize=15)
    fntmp = '%s/%strain_error' % (outPath, seedNo)
    DNN_tools.mySaveFig(plt, fntmp, ax=ax, isax=1, iseps=0)


def plotTrain_MSEs_RELs_3act_funcs(mse2data1, mse2data2, mse2data3, rel2data1, rel2data2, rel2data3, actName1=None,
                                   actName2=None, actName3=None, seedNo=1000, outPath=None, xaxis_scale=False,
                                   yaxis_scale=False):
    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    plt.plot(mse2data1, 'g-.', label=str('MSE-'+actName1))
    plt.plot(rel2data1, 'b:', label=str('REL-'+actName1))
    plt.plot(mse2data2, 'm--.', label=str('MSE-'+actName2))
    plt.plot(rel2data2, 'c-*', label=str('REL-'+actName2))
    plt.plot(mse2data3, color='k', marker='v', label=str('MSE-'+actName3))
    plt.plot(rel2data3, color='gold', marker='x', label=str('REL-'+actName3))
    if xaxis_scale:
        ax.set_yscale('log')
    if yaxis_scale:
        ax.set_yscale('log')
    plt.xlabel('epoch', fontsize=14)
    plt.ylabel('error', fontsize=14)
    ax.legend(loc='center', bbox_to_anchor=(0.485, 1.055), ncol=3, fontsize=12)
    # plt.legend(fontsize=11)
    # plt.title(' train error', fontsize=15)
    fntmp = '%s/%strain_Errs' % (outPath, seedNo)
    DNN_tools.mySaveFig(plt, fntmp, ax=ax, isax=1, iseps=0)


# ------------------------------------ plot test results --------------------------------------------------
def plot_2TestMSEs(data2mse1, data2mse2, mseType1=None, mseType2=None, epoches=None, seedNo=1000, outPath=None,
                 xaxis_scale=False, yaxis_scale=False):
    plt.figure()
    ax = plt.gca()
    plt.plot(epoches, data2mse1, 'r-.', label=mseType1)
    plt.plot(epoches, data2mse2, 'b:', label=mseType2)
    plt.xlabel('epoch/1000', fontsize=18)
    # plt.ylabel('L2error', fontsize=18)
    if xaxis_scale:
        ax.set_yscale('log')
    if yaxis_scale:
        ax.set_yscale('log')
    plt.legend(fontsize=18)
    plt.title('testing mse ', fontsize=15)
    fntmp = '%s/%stest_mse' % (outPath, seedNo)
    DNN_tools.mySaveFig(plt, fntmp, ax=ax, isax=1, iseps=0)


def plot_2TestRELs(data2rel1, data2rel2, relType1=None, relType2=None, epoches=1000, seedNo=1000, outPath=None,
                 xaxis_scale=False, yaxis_scale=False):
    plt.figure()
    ax = plt.gca()
    plt.plot(epoches, data2rel1, 'r-.', label=relType1)
    plt.plot(epoches, data2rel2, 'b:', label=relType2)
    plt.xlabel('epoch/1000', fontsize=18)
    # plt.ylabel('L2error', fontsize=18)
    if xaxis_scale:
        ax.set_yscale('log')
    if yaxis_scale:
        ax.set_yscale('log')
    plt.legend(fontsize=18)
    plt.title('testing mse ', fontsize=15)
    fntmp = '%s/%stest_rel' % (outPath, seedNo)
    DNN_tools.mySaveFig(plt, fntmp, ax=ax, isax=1, iseps=0)


def plotTest_MSE_REL(data2mse, data2rel, epoches, actName=None, seedNo=1000, outPath=None, xaxis_scale=False, yaxis_scale=False):
    plt.figure()
    ax = plt.gca()
    plt.plot(epoches, data2mse, 'r-.', label='MSE')
    plt.plot(epoches, data2rel, 'b:', label='REL')
    plt.xlabel('epoch/1000', fontsize=18)
    # plt.ylabel('L2error', fontsize=18)
    if xaxis_scale:
        ax.set_yscale('log')
    if yaxis_scale:
        ax.set_yscale('log')
    plt.legend(fontsize=18)
    plt.title('testing error ', fontsize=15)
    fntmp = '%s/%stestERR_%s' % (outPath, seedNo, str.lower(actName))
    DNN_tools.mySaveFig(plt, fntmp, ax=ax, isax=1, iseps=0)


def plot_Test_MSE_REL_2ActFuncs(data_mse1, data_rel1, data_mse2, data_rel2, epoches, actName1=None, actName2=None,
                                seedNo=1000, outPath=None, xaxis_scale=False, yaxis_scale=False):
    # fig2mse_test = plt.figure(figsize=(10, 8), dpi=98)
    fig2mse_test = plt.figure(figsize=(9, 6.5), dpi=98)
    ax = plt.gca()
    ax.plot(epoches, data_mse1, 'g-.', label=str('MSE-'+actName1))
    ax.plot(epoches, data_rel1, 'b:', label=str('REL'+actName1))
    ax.plot(epoches, data_mse2, 'm--', label=str('MSE'+actName2))
    ax.plot(epoches, data_rel2, 'c-*', label=str('REL'+actName2))
    plt.xlabel('epoch/1000', fontsize=14)
    plt.ylabel('error', fontsize=14)
    if xaxis_scale:
        ax.set_yscale('log')
    if yaxis_scale:
        ax.set_yscale('log')
    ax.legend(loc='center', bbox_to_anchor=(0.49, 1.06), ncol=3, fontsize=12)
    # plt.legend(fontsize=11)
    # plt.title('testing error ', fontsize=15)
    fntmp = '%s/%stest_error' % (outPath, seedNo)
    DNN_tools.mySaveFig(plt, fntmp, ax=ax, isax=1, iseps=0)


def plot_Test_MSE_REL_3Types(mse2s2ReLU, mse2sReLU, mse2ReLU, rel2s2ReLU, rel2sReLU, rel2ReLU, epoches=100,
                             seedNo=1000, outPath=None):
    # fig2mse_test = plt.figure(figsize=(10, 8), dpi=98)
    fig2mse_test = plt.figure(figsize=(9, 6.5), dpi=98)
    ax = plt.gca()
    ax.plot(epoches, mse2s2ReLU, 'g-.', label='MSE-s2ReLU')
    ax.plot(epoches, rel2s2ReLU, 'b:', label='REL-s2ReLU')
    ax.plot(epoches, mse2sReLU, 'm--', label='MSE-sReLU')
    ax.plot(epoches, rel2sReLU, 'c-*', label='REL-sReLU')
    ax.plot(epoches, mse2ReLU, color='k', marker='v', label='MSE-ReLU')
    ax.plot(epoches, rel2ReLU, color='gold', marker='x', label='REL-ReLU')
    plt.xlabel('epoch/1000', fontsize=14)
    plt.ylabel('error', fontsize=14)
    ax.set_yscale('log')
    ax.legend(loc='center', bbox_to_anchor=(0.49, 1.06), ncol=3, fontsize=12)
    # plt.legend(fontsize=11)
    # plt.title('testing error ', fontsize=15)
    fntmp = '%s/%stest_error' % (outPath, seedNo)
    DNN_tools.mySaveFig(plt, fntmp, ax=ax, isax=1, iseps=0)


def plotTest_MSEs_RELs_3act_funcs(mse2data1, mse2data2, mse2data3, rel2data1, rel2data2, rel2data3, actName1=None,
                                   actName2=None, actName3=None, seedNo=1000, outPath=None, xaxis_scale=False,
                                   yaxis_scale=False):
    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    plt.plot(mse2data1, 'g-.', label=str('MSE-'+actName1))
    plt.plot(rel2data1, 'b:', label=str('REL-'+actName1))
    plt.plot(mse2data2, 'm--.', label=str('MSE-'+actName2))
    plt.plot(rel2data2, 'c-*', label=str('REL-'+actName2))
    plt.plot(mse2data3, color='k', marker='v', label=str('MSE-'+actName3))
    plt.plot(rel2data3, color='gold', marker='x', label=str('REL-'+actName3))
    if xaxis_scale:
        ax.set_yscale('log')
    if yaxis_scale:
        ax.set_yscale('log')
    plt.xlabel('epoch', fontsize=14)
    plt.ylabel('error', fontsize=14)
    ax.legend(loc='center', bbox_to_anchor=(0.485, 1.055), ncol=3, fontsize=15)
    # plt.legend(fontsize=11)
    # plt.title(' train error', fontsize=15)
    fntmp = '%s/%stest_Errs' % (outPath, seedNo)
    DNN_tools.mySaveFig(plt, fntmp, ax=ax, isax=1, iseps=0)


def plot_2solutions2test(exact_solu2test, predict_solu2test,  coord_points2test=None,
                         batch_size2test=1000, seedNo=1000, outPath=None, subfig_type=1):
    if subfig_type == 1:
        plt.figure(figsize=(16, 10), dpi=98)
        fig, ax = plt.subplots(1, 1)  # fig, ax = plt.subplots(a,b)用来控制子图个数：a为行数，b为列数。
        ax.plot(coord_points2test, exact_solu2test, 'b-.', label='true')
        ax.plot(coord_points2test, predict_solu2test, 'g:', label='predict')
        ax.legend(fontsize=10)
        ax.set_xlabel('epoch', fontsize=18)

        axins = inset_axes(ax, width="50%", height="40%", loc=8, bbox_to_anchor=(0.2, 0.4, 0.5, 0.5),
                           bbox_transform=ax.transAxes)

        # 在子坐标系中绘制原始数据
        axins.plot(coord_points2test, exact_solu2test, color='b', linestyle='-.')

        axins.plot(coord_points2test, predict_solu2test, color='g', linestyle=':')

        axins.set_xticks([])
        axins.set_yticks([])

        # 设置放大区间
        zone_left = int(0.4 * batch_size2test)
        zone_right = int(0.4 * batch_size2test) + 100

        # 坐标轴的扩展比例（根据实际数据调整）
        x_ratio = 0.0  # x轴显示范围的扩展比例
        y_ratio = 0.075  # y轴显示范围的扩展比例

        # X轴的显示范围
        xlim0 = coord_points2test[zone_left] - (coord_points2test[zone_right] - coord_points2test[zone_left]) * x_ratio
        xlim1 = coord_points2test[zone_right] + (coord_points2test[zone_right] - coord_points2test[zone_left]) * x_ratio

        # Y轴的显示范围
        y = np.hstack((exact_solu2test[zone_left:zone_right], predict_solu2test[zone_left:zone_right]))
        ylim0 = np.min(y) - (np.max(y) - np.min(y)) * y_ratio
        ylim1 = np.max(y) + (np.max(y) - np.min(y)) * y_ratio

        # 调整子坐标系的显示范围
        axins.set_xlim(xlim0, xlim1)
        axins.set_ylim(ylim0, ylim1)

        # 建立父坐标系与子坐标系的连接线
        # loc1 loc2: 坐标系的四个角
        # 1 (右上) 2 (左上) 3(左下) 4(右下)
        mark_inset(ax, axins, loc1=3, loc2=1, fc="none", ec='k', lw=1)

        fntmp = '%s/%ssolu2test' % (outPath, seedNo)
        DNN_tools.mySaveFig(plt, fntmp, ax=ax, isax=1, iseps=0)
    elif subfig_type == 2:
        plt.figure(figsize=(16, 10), dpi=98)
        ax = plt.gca()
        p1 = plt.subplot(121)  # 1行2列，第一个图
        p2 = plt.subplot(122)  # 1行2列，第二个图

        p1.plot(coord_points2test, exact_solu2test, color='b', linestyle='-.', label='true')
        p1.plot(coord_points2test, predict_solu2test, color='g', linestyle=':', label='predict')
        ax.legend(fontsize=10)

        p2.plot(coord_points2test, exact_solu2test, color='b', linestyle='-.', label='true')
        p2.plot(coord_points2test, predict_solu2test, color='g', linestyle=':', label='predict')
        p2.axis([0.35, 0.65, 0.2, 0.27])

        # plot the box of
        tx0 = 0.35
        tx1 = 0.65
        ty0 = 0.2
        ty1 = 0.27
        sx = [tx0, tx1, tx1, tx0, tx0]
        sy = [ty0, ty0, ty1, ty1, ty0]
        p1.plot(sx, sy, "purple")

        # plot patch lines
        xy = (0.64, 0.265)
        xy2 = (0.36, 0.265)
        con = ConnectionPatch(xyA=xy2, xyB=xy, coordsA="data", coordsB="data", axesA=p2, axesB=p1)
        p2.add_artist(con)

        xy = (0.64, 0.21)
        xy2 = (0.36, 0.205)
        con = ConnectionPatch(xyA=xy2, xyB=xy, coordsA="data", coordsB="data",
                              axesA=p2, axesB=p1)
        p2.add_artist(con)

        fntmp = '%s/%ssolu2test' % (outPath, seedNo)
        DNN_tools.mySaveFig(plt, fntmp, ax=ax, isax=1, iseps=0)
    else:
        # fig11 = plt.figure(figsize=(10, 8))
        fig11 = plt.figure(figsize=(9, 6.5))
        ax = plt.gca()
        ax.plot(coord_points2test, exact_solu2test, 'b-.', label='exact')
        ax.plot(coord_points2test, predict_solu2test, 'r:', label='s2ReLU')
        # box = ax.get_position()
        # ax.set_position([box.x0, box.y0, box.width, box.height * 0.8])
        ax.legend(loc='right', bbox_to_anchor=(0.9, 1.05), ncol=4, fontsize=12)
        ax.set_xlabel('x', fontsize=14)
        ax.set_ylabel('u', fontsize=14)
        fntmp = '%s/%ssolu2test' % (outPath, seedNo)
        DNN_tools.mySaveFig(plt, fntmp, ax=ax, isax=1, iseps=0)


def plot_3solutions2test(exact_solu2test, s2ReLU_solu2test, sReLU_solu2test, ReLU_solu2test,
                         coord_points2test=None, batch_size2test=1000, seedNo=1000, outPath=None, subfig_type=1):
    # 嵌入绘制局部放大图的坐标系
    if subfig_type == 1:
        subgfig = plt.figure(figsize=(10, 8), dpi=98)
        ax = plt.gca()  # fig, ax = plt.subplots(a,b)用来控制子图个数：a为行数，b为列数。
        ax.plot(coord_points2test, exact_solu2test, 'b-.', label='exact')
        ax.plot(coord_points2test, s2ReLU_solu2test, 'g:', label='s2ReLU')
        ax.plot(coord_points2test, sReLU_solu2test, 'm--', label='sReLU')
        ax.plot(coord_points2test, ReLU_solu2test, 'c-', label='ReLU')
        ax.legend(loc='right', bbox_to_anchor=(0.85, 1.03), ncol=4, fontsize=12)
        ax.set_xlabel('epoch', fontsize=14)

        axins = inset_axes(ax, width="50%", height="40%", loc=8, bbox_to_anchor=(0.2, 0.2, 0.5, 0.5),
                           bbox_transform=ax.transAxes)

        # 在子坐标系中绘制原始数据
        axins.plot(coord_points2test, exact_solu2test, color='b', linestyle='-.')
        axins.plot(coord_points2test, s2ReLU_solu2test, color='g', linestyle=':')
        axins.plot(coord_points2test, sReLU_solu2test, color='m', linestyle='--')
        axins.plot(coord_points2test, ReLU_solu2test, color='c', linestyle='-')

        axins.set_xticks([])
        axins.set_yticks([])

        # 设置放大区间
        zone_left = int(0.4 * batch_size2test)
        zone_right = int(0.4 * batch_size2test) + 150

        # 坐标轴的扩展比例（根据实际数据调整）
        x_ratio = 0.075  # x轴显示范围的扩展比例
        y_ratio = 0.04  # y轴显示范围的扩展比例

        # X轴的显示范围
        xlim0 = coord_points2test[zone_left] - (coord_points2test[zone_right] - coord_points2test[zone_left]) * x_ratio
        xlim1 = coord_points2test[zone_right] + (coord_points2test[zone_right] - coord_points2test[zone_left]) * x_ratio

        # Y轴的显示范围
        y = np.hstack((exact_solu2test[zone_left:zone_right], s2ReLU_solu2test[zone_left:zone_right]))
        ylim0 = np.min(y) - (np.max(y) - np.min(y)) * y_ratio
        ylim1 = np.max(y) + (np.max(y) - np.min(y)) * y_ratio

        # 调整子坐标系的显示范围
        axins.set_xlim(xlim0, xlim1)
        axins.set_ylim(ylim0, ylim1)

        # 建立父坐标系与子坐标系的连接线
        # loc1 loc2: 坐标系的四个角
        # 1 (右上) 2 (左上) 3(左下) 4(右下)
        mark_inset(ax, axins, loc1=3, loc2=1, fc="none", ec='k', lw=1)

        fntmp = '%s/%ssolu2test' % (outPath, seedNo)
        DNN_tools.mySaveFig(plt, fntmp, ax=ax, isax=1, iseps=0)
    elif subfig_type == 2:
        plt.figure(figsize=(16, 10), dpi=98)
        ax = plt.gca()
        p1 = plt.subplot(121)  # 1行2列，第一个图
        p2 = plt.subplot(122)  # 1行2列，第二个图

        p1.plot(coord_points2test, exact_solu2test, color='b', linestyle='-.', label='true')
        p1.plot(coord_points2test, s2ReLU_solu2test, color='g', linestyle=':', label='predict')
        ax.legend(fontsize=10)

        p2.plot(coord_points2test, exact_solu2test, color='b', linestyle='-.', label='true')
        p2.plot(coord_points2test, s2ReLU_solu2test, color='g', linestyle=':', label='predict')
        p2.axis([0.35, 0.65, 0.2, 0.27])

        # plot the box of
        tx0 = 0.35
        tx1 = 0.65
        ty0 = 0.2
        ty1 = 0.27
        sx = [tx0, tx1, tx1, tx0, tx0]
        sy = [ty0, ty0, ty1, ty1, ty0]
        p1.plot(sx, sy, "purple")

        # plot patch lines
        xy = (0.64, 0.265)
        xy2 = (0.36, 0.265)
        con = ConnectionPatch(xyA=xy2, xyB=xy, coordsA="data", coordsB="data", axesA=p2, axesB=p1)
        p2.add_artist(con)

        xy = (0.64, 0.21)
        xy2 = (0.36, 0.205)
        con = ConnectionPatch(xyA=xy2, xyB=xy, coordsA="data", coordsB="data",
                              axesA=p2, axesB=p1)
        p2.add_artist(con)

        fntmp = '%s/%ssolu2test' % (outPath, seedNo)
        DNN_tools.mySaveFig(plt, fntmp, ax=ax, isax=1, iseps=0)
    else:
        # fig11 = plt.figure(figsize=(10, 8))
        fig11 = plt.figure(figsize=(9, 6.5))
        ax = plt.gca()
        ax.plot(coord_points2test, exact_solu2test, 'b-.', label='exact')
        ax.plot(coord_points2test, s2ReLU_solu2test, 'g:', label='s2ReLU')
        ax.plot(coord_points2test, sReLU_solu2test, 'm--', label='sReLU')
        ax.plot(coord_points2test, ReLU_solu2test, 'c-', label='ReLU')
        # box = ax.get_position()
        # ax.set_position([box.x0, box.y0, box.width, box.height * 0.8])
        ax.legend(loc='right', bbox_to_anchor=(0.9, 1.05), ncol=4, fontsize=12)
        ax.set_xlabel('x', fontsize=14)
        ax.set_ylabel('u', fontsize=14)
        fntmp = '%s/%ssolu2test' % (outPath, seedNo)
        DNN_tools.mySaveFig(plt, fntmp, ax=ax, isax=1, iseps=0)


def plot_Hot_solution2test(solu2test, size_vec2mat=20, actName=None, seedNo=1000, outPath=None):
    solu2color = np.reshape(solu2test, (size_vec2mat, size_vec2mat))
    plt.figure()
    ax = plt.gca()
    plt.imshow(solu2color, interpolation='nearest', cmap=cm.coolwarm, origin='lower')
    plt.colorbar(shrink=0.9)
    plt.xticks(())
    plt.yticks(())
    # plt.title('exact solution', fontsize=14)
    if str.lower(actName) == 'utrue':
        fntmp = '%s/%s%s' % (outPath, seedNo, 'Utrue2test')
    elif str.lower(actName) == 'srelu':
        fntmp = '%s/%s_%s' % (outPath, seedNo, 'UsReLU2test')
    elif str.lower(actName) == 's2relu':
        fntmp = '%s/%s_%s' % (outPath, seedNo, 'Us2ReLU2test')
    elif str.lower(actName) == 's3relu':
        fntmp = '%s/%s_%s' % (outPath, seedNo, 'Us3ReLU2test')
    elif str.lower(actName) == 'csrelu':
        fntmp = '%s/%s_%s' % (outPath, seedNo, 'UCsReLU2test')
    elif str.lower(actName) == 'relu':
        fntmp = '%s/%s_%s' % (outPath, seedNo, 'UReLU2test')
    elif str.lower(actName) == 'tanh':
        fntmp = '%s/%s_%s' % (outPath, seedNo, 'Utanh2test')
    elif str.lower(actName) == 'sintanh':
        fntmp = '%s/%s_%s' % (outPath, seedNo, 'Ustanh2test')
    elif str.lower(actName) == 'singauss':
        fntmp = '%s/%s_%s' % (outPath, seedNo, 'Usgauss2test')
    elif str.lower(actName) == 'gauss':
        fntmp = '%s/%s_%s' % (outPath, seedNo, 'Ugauss2test')
    elif str.lower(actName) == 'mexican':
        fntmp = '%s/%s_%s' % (outPath, seedNo, 'Umexican2test')
    elif str.lower(actName) == 'modify_mexican':
        fntmp = '%s/%s_%s' % (outPath, seedNo, 'Ummexican2test')
    elif str.lower(actName) == 'sin_modify_mexican':
        fntmp = '%s/%s_%s' % (outPath, seedNo, 'Usm-mexican2test')
    else:
        fntmp = '%s/U%s' % (outPath, str.lower(actName))
    DNN_tools.mySaveFig(plt, fntmp, ax=ax, isax=1, iseps=0)


def plot_scatter_solution2test(solu2test, test_batch, actName=None, seedNo=1000, outPath=None):
    dim2test_batch = 2
    if 2 == dim2test_batch:
        test_x_bach = np.reshape(test_batch[:, 0], newshape=[-1, 1])
        test_y_bach = np.reshape(test_batch[:, 1], newshape=[-1, 1])

        # 绘制解的3D散点图
        fig = plt.figure(figsize=(10, 10))
        ax = Axes3D(fig)
        ax.scatter(test_x_bach, test_y_bach, solu2test, c='b', label=actName)

        # 绘制图例
        ax.legend(loc='best')
        # 添加坐标轴(顺序是X，Y, Z)
        ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})
        ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
        ax.set_zlabel('u', fontdict={'size': 15, 'color': 'red'})

        # plt.title('solution', fontsize=15)
        fntmp = '%s/%ssolu' % (outPath, seedNo)
        DNN_tools.mySaveFig(plt, fntmp, ax=ax, isax=1, iseps=0)
    else:
        return


def plot_scatter_solutions2test(solu1_test, solu2_test, test_batch, actName1=None, actName2=None, seedNo=1000,
                                outPath=None):
    dim2test_batch = 2
    if 2 == dim2test_batch:
        test_x_bach = np.reshape(test_batch[:, 0], newshape=[-1, 1])
        test_y_bach = np.reshape(test_batch[:, 1], newshape=[-1, 1])

        # 绘制解的3D散点图(真解和预测解)
        fig = plt.figure(figsize=(10, 10))
        ax = Axes3D(fig)
        ax.scatter(test_x_bach, test_y_bach, solu1_test, c='b', label=actName1)
        ax.scatter(test_x_bach, test_y_bach, solu2_test, c='b', label=actName2)

        # 绘制图例
        ax.legend(loc='best')
        # 添加坐标轴(顺序是X，Y, Z)
        ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})
        ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
        ax.set_zlabel('u', fontdict={'size': 15, 'color': 'red'})

        # plt.title('solution', fontsize=15)
        fntmp = '%s/%ssolus_%s' % (outPath, seedNo, actName2)
        DNN_tools.mySaveFig(plt, fntmp, ax=ax, isax=1, iseps=0)
    else:
        return


def plot_Hot_point_wise_err(point_wise_err, size_vec2mat=20, actName=None, seedNo=1000, outPath=None):
    # 逐点误差分布热力图
    square_err_color2sin = np.reshape(point_wise_err, (size_vec2mat, size_vec2mat))
    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    plt.imshow(square_err_color2sin, interpolation='nearest', cmap=cm.coolwarm, origin='lower')
    plt.colorbar(shrink=0.85)
    plt.xticks(())
    plt.yticks(())
    # plt.title('point-wise error', fontsize=14)
    if str.lower(actName) == 'srelu':
        fntmp = '%s/%spErr_%s' % (outPath, seedNo, 'sReLU')
    elif str.lower(actName) == 's2relu':
        fntmp = '%s/%spErr_%s' % (outPath, seedNo, 's2ReLU')
    elif str.lower(actName) == 's3relu':
        fntmp = '%s/%spErr_%s' % (outPath, seedNo, 's3ReLU')
    elif str.lower(actName) == 'csrelu':
        fntmp = '%s/%spErr_%s' % (outPath, seedNo, 'CsReLU')
    elif str.lower(actName) == 'relu':
        fntmp = '%s/%spErr_%s' % (outPath, seedNo, 'ReLU')
    elif str.lower(actName) == 'tanh':
        fntmp = '%s/%spErr_%s' % (outPath, seedNo, 'tanh')
    elif str.lower(actName) == 'sin':
        fntmp = '%s/%spErr_%s' % (outPath, seedNo, 'sin')
    elif str.lower(actName) == 'sintanh':
        fntmp = '%s/%spErr_%s' % (outPath, seedNo, 'stanh')
    elif str.lower(actName) == 'singauss':
        fntmp = '%s/%spErr_%s' % (outPath, seedNo, 'sgauss')
    elif str.lower(actName) == 'gauss':
        fntmp = '%s/%spErr_%s' % (outPath, seedNo, 'gauss')
    elif str.lower(actName) == 'mexican':
        fntmp = '%s/%spErr_%s' % (outPath, seedNo, 'mexican')
    elif str.lower(actName) == 'modify_mexican':
        fntmp = '%s/%spErr_%s' % (outPath, seedNo, 'mmexican')
    elif str.lower(actName) == 'sin_modify_mexican':
        fntmp = '%s/%spErr_%s' % (outPath, seedNo, 'sm-mexican')
    else:
        fntmp = '%s/pErr_%s' % (outPath, str.lower(actName))
    DNN_tools.mySaveFig(plt, fntmp, ax=ax, isax=1, iseps=0)


