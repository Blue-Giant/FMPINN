import numpy as np
import torch
import scipy.stats.qmc as stqmc
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D


# ---------------------------------------------- 数据集的生成 ---------------------------------------------------
#  内部生成，方形区域[a,b]^n生成随机数, n代表变量个数，使用随机采样方法
def rand_it(batch_size, variable_dim, region_a, region_b, to_torch=True, to_float=True, to_cuda=False, gpu_no=0,
            use_grad=False):
    # np.random.rand( )可以返回一个或一组服从“0~1”均匀分布的随机样本值。随机样本取值范围是[0,1)，不包括1。
    # np.random.rand(3,2 )可以返回一个或一组服从“0~1”均匀分布的随机矩阵(3行2列)。随机样本取值范围是[0,1)，不包括1。
    x_it = (region_b - region_a) * np.random.rand(batch_size, variable_dim) + region_a
    if to_float:
        x_it = x_it.astype(np.float32)

    if to_torch:
        x_it = torch.from_numpy(x_it)

        if to_cuda:
            x_it = x_it.cuda(device='cuda:' + str(gpu_no))

        x_it.requires_grad = use_grad

    return x_it


#  内部生成,矩形区域
def rand_in_1D(batch_size=100, variable_dim=1, region_a=0.0, region_b=1.0, to_torch=True, to_float=True,
               to_cuda=False, gpu_no=0, use_grad=False, opt2sampler='lhs'):
    # np.random.rand( )可以返回一个或一组服从“0~1”均匀分布的随机样本值。随机样本取值范围是[0,1)，不包括1。
    # np.random.rand(3,2 )可以返回一个或一组服从“0~1”均匀分布的随机矩阵(3行2列)。随机样本取值范围是[0,1)，不包括1。
    assert 1 == int(variable_dim)
    if 'lhs' == opt2sampler:
        sampler = stqmc.LatinHypercube(d=variable_dim)
        x_it = (region_b - region_a) * sampler.random(batch_size) + region_a
    else:
        x_it = (region_b - region_a) * np.random.rand(batch_size, 1) + region_a

    if to_float:
        x_it = x_it.astype(np.float32)

    if to_torch:
        x_it = torch.from_numpy(x_it)

        if to_cuda:
            x_it = x_it.cuda(device='cuda:' + str(gpu_no))

        x_it.requires_grad = use_grad

    return x_it


#  内部生成, 矩形区域, 使用随机采样方法
def rand_in_2D(batch_size=100, variable_dim=2, region_left=0.0, region_right=1.0, region_bottom=0.0, region_top=1.0,
               to_torch=True, to_float=True, to_cuda=False, gpu_no=0, use_grad=False, opt2sampler='lhs'):
    # np.random.rand( )可以返回一个或一组服从“0~1”均匀分布的随机样本值。随机样本取值范围是[0,1)，不包括1。
    # np.random.rand(3,2 )可以返回一个或一组服从“0~1”均匀分布的随机矩阵(3行2列)。随机样本取值范围是[0,1)，不包括1。
    assert 2 == int(variable_dim)
    if 'lhs' == opt2sampler:
        sampler = stqmc.LatinHypercube(d=1)
        x_it = (region_right - region_left) * sampler.random(batch_size) + region_left
        y_it = (region_top - region_bottom) * sampler.random(batch_size) + region_bottom
    else:
        x_it = (region_right - region_left) * np.random.rand(batch_size, 1) + region_left
        y_it = (region_top - region_bottom) * np.random.rand(batch_size, 1) + region_bottom

    xy_in = np.concatenate([x_it, y_it], axis=-1)
    if to_float:
        xy_in = xy_in.astype(np.float32)

    if to_torch:
        xy_in = torch.from_numpy(xy_in)

        if to_cuda:
            xy_in = xy_in.cuda(device='cuda:' + str(gpu_no))

        xy_in.requires_grad = use_grad

    return xy_in


#  内部生成, 矩形区域, 使用随机采样方法
def rand_in_3D(batch_size=100, variable_dim=3, region_left=0.0, region_right=1.0, region_behind=0.0, region_front=1.0,
               region_bottom=0.0, region_top=1.0, to_torch=True, to_float=True, to_cuda=False, gpu_no=0,
               use_grad=False, opt2sampler='lhs'):
    # np.random.rand( )可以返回一个或一组服从“0~1”均匀分布的随机样本值。随机样本取值范围是[0,1)，不包括1。
    # np.random.rand(3,2 )可以返回一个或一组服从“0~1”均匀分布的随机矩阵(3行2列)。随机样本取值范围是[0,1)，不包括1。
    assert 3 == int(variable_dim)
    if 'lhs' == opt2sampler:
        sampler = stqmc.LatinHypercube(d=1)
        x_it = (region_right - region_left) * sampler.random(batch_size) + region_left
        y_it = (region_front - region_behind) * sampler.random(batch_size) + region_behind
        z_it = (region_top - region_bottom) * sampler.random(batch_size) + region_bottom
    else:
        x_it = (region_right - region_left) * np.random.rand(batch_size, 1) + region_left
        y_it = (region_front - region_behind) * np.random.rand(batch_size, 1) + region_behind
        z_it = (region_top - region_bottom) * np.random.rand(batch_size, 1) + region_bottom

    xyz_in = np.concatenate([x_it, y_it, z_it], axis=-1)
    if to_float:
        xyz_in = xyz_in.astype(np.float32)

    if to_torch:
        xyz_in = torch.from_numpy(xyz_in)

        if to_cuda:
            xyz_in = xyz_in.cuda(device='cuda:' + str(gpu_no))

        xyz_in.requires_grad = use_grad

    return xyz_in


#  内部生成,矩形区域, 使用随机采样方法
def rand_in_4D(batch_size=100, variable_dim=3, region_xleft=0.0, region_xright=1.0, region_yleft=0.0, region_yright=1.0,
               region_zleft=0.0, region_zright=1.0, region_sleft=0.0, region_sright=1.0, to_torch=True, to_float=True,
               to_cuda=False, gpu_no=0, use_grad=False, opt2sampler='lhs'):
    # np.random.rand( )可以返回一个或一组服从“0~1”均匀分布的随机样本值。随机样本取值范围是[0,1)，不包括1。
    # np.random.rand(3,2 )可以返回一个或一组服从“0~1”均匀分布的随机矩阵(3行2列)。随机样本取值范围是[0,1)，不包括1。
    assert 4 == int(variable_dim)
    if 'lhs' == opt2sampler:
        sampler = stqmc.LatinHypercube(d=1)
        x_it = (region_xright - region_xleft) * sampler.random(batch_size) + region_xleft
        y_it = (region_yright - region_yleft) * sampler.random(batch_size) + region_yleft
        z_it = (region_zright - region_zleft) * sampler.random(batch_size) + region_zleft
        s_it = (region_sright - region_sleft) * sampler.random(batch_size) + region_sleft
    else:
        x_it = (region_xright - region_xleft) * np.random.rand(batch_size, 1) + region_xleft
        y_it = (region_yright - region_yleft) * np.random.rand(batch_size, 1) + region_yleft
        z_it = (region_zright - region_zleft) * np.random.rand(batch_size, 1) + region_zleft
        s_it = (region_sright - region_sleft) * np.random.rand(batch_size, 1) + region_sleft

    xyzs_in = np.concatenate([x_it, y_it, z_it, s_it], axis=-1)
    if to_float:
        xyzs_in = xyzs_in.astype(np.float32)

    if to_torch:
        xyzs_in = torch.from_numpy(xyzs_in)

        if to_cuda:
            xyzs_in = xyzs_in.cuda(device='cuda:' + str(gpu_no))

        xyzs_in.requires_grad = use_grad

    return xyzs_in


#  内部生成,矩形区域, 使用随机采样方法
def rand_in_5D(batch_size=100, variable_dim=3, region_xleft=0.0, region_xright=1.0, region_yleft=0.0, region_yright=1.0,
               region_zleft=0.0, region_zright=1.0, region_sleft=0.0, region_sright=1.0, region_tleft=0.0,
               region_tright=1.0, to_torch=True, to_float=True, to_cuda=False, gpu_no=0, use_grad=False,
               opt2sampler='lhs'):
    # np.random.rand( )可以返回一个或一组服从“0~1”均匀分布的随机样本值。随机样本取值范围是[0,1)，不包括1。
    # np.random.rand(3,2 )可以返回一个或一组服从“0~1”均匀分布的随机矩阵(3行2列)。随机样本取值范围是[0,1)，不包括1。
    assert 5 == int(variable_dim)
    if 'lhs' == opt2sampler:
        sampler = stqmc.LatinHypercube(d=1)
        x_it = (region_xright - region_xleft) * sampler.random(batch_size) + region_xleft
        y_it = (region_yright - region_yleft) * sampler.random(batch_size) + region_yleft
        z_it = (region_zright - region_zleft) * sampler.random(batch_size) + region_zleft
        s_it = (region_sright - region_sleft) * sampler.random(batch_size) + region_sleft
        t_it = (region_tright - region_tleft) * sampler.random(batch_size) + region_tleft
    else:
        x_it = (region_xright - region_xleft) * np.random.rand(batch_size, 1) + region_xleft
        y_it = (region_yright - region_yleft) * np.random.rand(batch_size, 1) + region_yleft
        z_it = (region_zright - region_zleft) * np.random.rand(batch_size, 1) + region_zleft
        s_it = (region_sright - region_sleft) * np.random.rand(batch_size, 1) + region_sleft
        t_it = (region_tright - region_tleft) * np.random.rand(batch_size, 1) + region_tleft

    xyzst_in = np.concatenate([x_it, y_it, z_it, s_it, t_it], axis=-1)
    if to_float:
        xyzst_in = xyzst_in.astype(np.float32)

    if to_torch:
        xyzst_in = torch.from_numpy(xyzst_in)

        if to_cuda:
            xyzst_in = xyzst_in.cuda(device='cuda:' + str(gpu_no))

        xyzst_in.requires_grad = use_grad

    return xyzst_in


#  内部生成,矩形区域, 使用随机采样方法
def rand_in_8D(batch_size=100, variable_dim=3, region_xleft=0.0, region_xright=1.0, region_yleft=0.0, region_yright=1.0,
               region_zleft=0.0, region_zright=1.0, region_sleft=0.0, region_sright=1.0, region_tleft=0.0,
               region_tright=1.0, region_pleft=0.0, region_pright=1.0, region_qleft=0.0, region_qright=1.0,
               region_rleft=0.0, region_rright=1.0, to_torch=True, to_float=True, to_cuda=False, gpu_no=0,
               use_grad=False, opt2sampler='lhs'):
    # np.random.rand( )可以返回一个或一组服从“0~1”均匀分布的随机样本值。随机样本取值范围是[0,1)，不包括1。
    # np.random.rand(3,2 )可以返回一个或一组服从“0~1”均匀分布的随机矩阵(3行2列)。随机样本取值范围是[0,1)，不包括1。
    assert 8 == int(variable_dim)
    if 'lhs' == opt2sampler:
        sampler = stqmc.LatinHypercube(d=1)
        x_it = (region_xright - region_xleft) * sampler.random(batch_size) + region_xleft
        y_it = (region_yright - region_yleft) * sampler.random(batch_size) + region_yleft
        z_it = (region_zright - region_zleft) * sampler.random(batch_size) + region_zleft
        s_it = (region_sright - region_sleft) * sampler.random(batch_size) + region_sleft
        t_it = (region_tright - region_tleft) * sampler.random(batch_size) + region_tleft
        p_it = (region_pright - region_pleft) * sampler.random(batch_size) + region_pleft
        q_it = (region_qright - region_qleft) * sampler.random(batch_size) + region_qleft
        r_it = (region_rright - region_rleft) * sampler.random(batch_size) + region_rleft
    else:
        x_it = (region_xright - region_xleft) * np.random.rand(batch_size, 1) + region_xleft
        y_it = (region_yright - region_yleft) * np.random.rand(batch_size, 1) + region_yleft
        z_it = (region_zright - region_zleft) * np.random.rand(batch_size, 1) + region_zleft
        s_it = (region_sright - region_sleft) * np.random.rand(batch_size, 1) + region_sleft
        t_it = (region_tright - region_tleft) * np.random.rand(batch_size, 1) + region_tleft
        p_it = (region_pright - region_pleft) * np.random.rand(batch_size, 1) + region_pleft
        q_it = (region_qright - region_qleft) * np.random.rand(batch_size, 1) + region_qleft
        r_it = (region_rright - region_rleft) * np.random.rand(batch_size, 1) + region_rleft

    xyzstpqr_in = np.concatenate([x_it, y_it, z_it, s_it, t_it, p_it, q_it, r_it], axis=-1)
    if to_float:
        xyzstpqr_in = xyzstpqr_in.astype(np.float32)

    if to_torch:
        xyzstpqr_in = torch.from_numpy(xyzstpqr_in)

        if to_cuda:
            xyzstpqr_in = xyzstpqr_in.cuda(device='cuda:' + str(gpu_no))

        xyzstpqr_in.requires_grad = use_grad

    return xyzstpqr_in


#  内部生成, 方形区域[a,b]^n生成随机数, n代表变量个数. 使用Sobol采样方法
def rand_it_Sobol(batch_size, variable_dim, region_a, region_b, to_torch=True, to_float=True, to_cuda=False, gpu_no=0,
                   use_grad2x=False):
    sampler = stqmc.Sobol(d=variable_dim, scramble=True)
    x_it = (region_b - region_a) * sampler.random(n=batch_size) + region_a
    if to_float:
        x_it = x_it.astype(np.float32)

    if to_torch:
        x_it = torch.from_numpy(x_it)

        if to_cuda:
            x_it = x_it.cuda(device='cuda:' + str(gpu_no))

        x_it.requires_grad = use_grad2x

    return x_it


# 边界生成点
def rand_bd_1D(batch_size=1000, variable_dim=1, region_a=0.0, region_b=1.0, to_torch=True, to_float=True,
               to_cuda=False, gpu_no=0, use_grad=False):
    # np.asarray 将输入转为矩阵格式。
    # 当输入是列表的时候，更改列表的值并不会影响转化为矩阵的值
    # [0,1] 转换为 矩阵，然后
    # reshape(-1,1):数组新的shape属性应该要与原来的配套，如果等于-1的话，那么Numpy会根据剩下的维度计算出数组的另外一个shape属性值。
    assert (int(variable_dim) == 1)

    # region_a = float(region_a)
    # region_b = float(region_b)

    x_left_bd = np.ones(shape=[batch_size, 1], dtype=np.float32) * region_a
    x_right_bd = np.ones(shape=[batch_size, 1], dtype=np.float32) * region_b
    if to_float:
        x_left_bd = x_left_bd.astype(np.float32)
        x_right_bd = x_right_bd.astype(np.float32)

    if to_torch:
        x_left_bd = torch.from_numpy(x_left_bd)
        x_right_bd = torch.from_numpy(x_right_bd)

        if to_cuda:
            x_left_bd = x_left_bd.cuda(device='cuda:' + str(gpu_no))
            x_right_bd = x_right_bd.cuda(device='cuda:' + str(gpu_no))

        x_left_bd.requires_grad = use_grad
        x_right_bd.requires_grad = use_grad

    return x_left_bd, x_right_bd


#  内部生成，方形区域[a,b]^n生成随机数, n代表变量个数，使用随机采样方法
def rand_bd_2D(batch_size=1000, variable_dim=2, region_left=0.0, region_right=1.0, region_bottom=0.0, region_top=1.0,
               to_torch=False, to_float=True, to_cuda=False, gpu_no=0, use_grad=False, opt2sampler='lhs'):
    # np.asarray 将输入转为矩阵格式。
    # 当输入是列表的时候，更改列表的值并不会影响转化为矩阵的值
    # [0,1] 转换为 矩阵，然后
    # reshape(-1,1):数组新的shape属性应该要与原来的配套，如果等于-1的话，那么Numpy会根据剩下的维度计算出数组的另外一个shape属性值。
    # np.random.random((100, 50)) 上方代表生成100行 50列的随机浮点数，浮点数范围 : (0,1)
    # np.random.random([100, 50]) 和 np.random.random((100, 50)) 效果一样

    assert (int(variable_dim) == 2)

    if 'lhs' == opt2sampler:
        sampler = stqmc.LatinHypercube(d=variable_dim)
        x_left_bd = (region_top - region_bottom) * sampler.random(batch_size) + region_bottom
        x_right_bd = (region_top - region_bottom) * sampler.random(batch_size) + region_bottom
        y_bottom_bd = (region_right - region_left) * sampler.random(batch_size) + region_left
        y_top_bd = (region_right - region_left) * sampler.random(batch_size) + region_left
    else:
        x_left_bd = (region_top - region_bottom) * np.random.random([batch_size, 2]) + region_bottom  # 浮点数都是从0-1中随机。
        x_right_bd = (region_top - region_bottom) * np.random.random([batch_size, 2]) + region_bottom
        y_bottom_bd = (region_right - region_left) * np.random.random([batch_size, 2]) + region_left
        y_top_bd = (region_right - region_left) * np.random.random([batch_size, 2]) + region_left

    for ii in range(batch_size):
        x_left_bd[ii, 0] = region_left
        x_right_bd[ii, 0] = region_right
        y_bottom_bd[ii, 1] = region_bottom
        y_top_bd[ii, 1] = region_top

    if to_float:
        x_left_bd = x_left_bd.astype(np.float32)
        x_right_bd = x_right_bd.astype(np.float32)
        y_bottom_bd = y_bottom_bd.astype(np.float32)
        y_top_bd = y_top_bd.astype(np.float32)
    if to_torch:
        x_left_bd = torch.from_numpy(x_left_bd)
        x_right_bd = torch.from_numpy(x_right_bd)
        y_bottom_bd = torch.from_numpy(y_bottom_bd)
        y_top_bd = torch.from_numpy(y_top_bd)
        if to_cuda:
            x_left_bd = x_left_bd.cuda(device='cuda:' + str(gpu_no))
            x_right_bd = x_right_bd.cuda(device='cuda:' + str(gpu_no))
            y_bottom_bd = y_bottom_bd.cuda(device='cuda:' + str(gpu_no))
            y_top_bd = y_top_bd.cuda(device='cuda:' + str(gpu_no))

        x_left_bd.requires_grad = use_grad
        x_right_bd.requires_grad = use_grad
        y_bottom_bd.requires_grad = use_grad
        y_top_bd.requires_grad = use_grad

    return x_left_bd, x_right_bd, y_bottom_bd, y_top_bd


#  内部生成, 方形区域[a,b]^n生成随机数, n代表变量个数. 使用Sobol采样方法
def rand_bd_2D_sobol(batch_size, variable_dim, region_a, region_b, to_torch=True, to_float=True, to_cuda=False,
                     gpu_no=0, use_grad=False):
    region_a = float(region_a)
    region_b = float(region_b)
    assert (int(variable_dim) == 2)
    sampler = stqmc.Sobol(d=variable_dim, scramble=True)
    x_left_bd = (region_b-region_a) * sampler.random(n=batch_size) + region_a   # 浮点数都是从0-1中随机。
    x_right_bd = (region_b - region_a) * sampler.random(n=batch_size) + region_a
    y_bottom_bd = (region_b - region_a) * sampler.random(n=batch_size) + region_a
    y_top_bd = (region_b - region_a) * sampler.random(n=batch_size) + region_a
    for ii in range(batch_size):
        x_left_bd[ii, 0] = region_a
        x_right_bd[ii, 0] = region_b
        y_bottom_bd[ii, 1] = region_a
        y_top_bd[ii, 1] = region_b

    if to_float:
        x_left_bd = x_left_bd.astype(np.float32)
        x_right_bd = x_right_bd.astype(np.float32)
        y_bottom_bd = y_bottom_bd.astype(np.float32)
        y_top_bd = y_top_bd.astype(np.float32)
    if to_torch:
        x_left_bd = torch.from_numpy(x_left_bd)
        x_right_bd = torch.from_numpy(x_right_bd)
        y_bottom_bd = torch.from_numpy(y_bottom_bd)
        y_top_bd = torch.from_numpy(y_top_bd)
        if to_cuda:
            x_left_bd = x_left_bd.cuda(device='cuda:' + str(gpu_no))
            x_right_bd = x_right_bd.cuda(device='cuda:' + str(gpu_no))
            y_bottom_bd = y_bottom_bd.cuda(device='cuda:' + str(gpu_no))
            y_top_bd = y_top_bd.cuda(device='cuda:' + str(gpu_no))

        x_left_bd.requires_grad = use_grad
        x_right_bd.requires_grad = use_grad
        y_bottom_bd.requires_grad = use_grad
        y_top_bd.requires_grad = use_grad

    return x_left_bd, x_right_bd, y_bottom_bd, y_top_bd


def rand_bd_3D(batch_size=1000, variable_dim=3, region_left=0.0, region_right=1.0, region_behind=0.0, region_front=1.0,
               region_bottom=0.0, region_top=1.0, to_torch=True, to_float=True, to_cuda=False, gpu_no=0,
               use_grad=False, opt2sampler='lhs'):
    # np.asarray 将输入转为矩阵格式。
    # 当输入是列表的时候，更改列表的值并不会影响转化为矩阵的值
    # [0,1] 转换为 矩阵，然后
    # reshape(-1,1):数组新的shape属性应该要与原来的配套，如果等于-1的话，那么Numpy会根据剩下的维度计算出数组的另外一个shape属性值。
    assert (int(variable_dim) == 3)
    x_left_bd = np.zeros(shape=[batch_size, variable_dim])
    x_right_bd = np.zeros(shape=[batch_size, variable_dim])
    y_behind_bd = np.zeros(shape=[batch_size, variable_dim])
    y_front_bd = np.zeros(shape=[batch_size, variable_dim])
    z_bottom_bd = np.zeros(shape=[batch_size, variable_dim])
    z_top_bd = np.zeros(shape=[batch_size, variable_dim])

    if 'lhs' == opt2sampler:
        sampler = stqmc.LatinHypercube(d=1)
        x_left_bd[:, 0:1] = region_left
        x_left_bd[:, 1:2] = (region_front - region_behind) * sampler.random(batch_size) + region_behind
        x_left_bd[:, 2:3] = (region_top - region_bottom) * sampler.random(batch_size) + region_bottom

        x_right_bd[:, 0:1] = region_right
        x_right_bd[:, 1:2] = (region_front - region_behind) * sampler.random(batch_size) + region_behind
        x_right_bd[:, 2:3] = (region_top - region_bottom) * sampler.random(batch_size) + region_bottom

        y_behind_bd[:, 0:1] = (region_right - region_left) * sampler.random(batch_size) + region_left
        y_behind_bd[:, 1:2] = region_behind
        y_behind_bd[:, 2:3] = (region_top - region_bottom) * sampler.random(batch_size) + region_bottom

        y_front_bd[:, 0:1] = (region_right - region_left) * sampler.random(batch_size) + region_left
        y_front_bd[:, 1:2] = region_front
        y_front_bd[:, 2:3] = (region_top - region_bottom) * sampler.random(batch_size) + region_bottom

        z_bottom_bd[:, 0:1] = (region_right - region_left) * sampler.random(batch_size) + region_left
        z_bottom_bd[:, 1:2] = (region_front - region_behind) * sampler.random(batch_size) + region_behind
        z_bottom_bd[:, 2:3] = region_bottom

        z_top_bd[:, 0:1] = (region_right - region_left) * sampler.random(batch_size) + region_left
        z_top_bd[:, 1:2] = (region_front - region_behind) * sampler.random(batch_size) + region_behind
        z_top_bd[:, 2:3] = region_top
    else:
        x_left_bd[:, 0:1] = region_left
        x_left_bd[:, 1:2] = (region_front - region_behind) * np.random.random([batch_size, 1]) + region_behind
        x_left_bd[:, 2:3] = (region_top - region_bottom) * np.random.random([batch_size, 1]) + region_bottom

        x_right_bd[:, 0:1] = region_right
        x_right_bd[:, 1:2] = (region_front - region_behind) * np.random.random([batch_size, 1]) + region_behind
        x_right_bd[:, 2:3] = (region_top - region_bottom) * np.random.random([batch_size, 1]) + region_bottom

        y_behind_bd[:, 0:1] = (region_right - region_left) * np.random.random([batch_size, 1]) + region_left
        y_behind_bd[:, 1:2] = region_behind
        y_behind_bd[:, 2:3] = (region_top - region_bottom) * np.random.random([batch_size, 1]) + region_bottom

        y_front_bd[:, 0:1] = (region_right - region_left) * np.random.random([batch_size, 1]) + region_left
        y_front_bd[:, 1:2] = region_front
        y_front_bd[:, 2:3] = (region_top - region_bottom) * np.random.random([batch_size, 1]) + region_bottom

        z_bottom_bd[:, 0:1] = (region_right - region_left) * np.random.random([batch_size, 1]) + region_left
        z_bottom_bd[:, 1:2] = (region_front - region_behind) * np.random.random([batch_size, 1]) + region_behind
        z_bottom_bd[:, 2:3] = region_bottom

        z_top_bd[:, 0:1] = (region_right - region_left) * np.random.random([batch_size, 1]) + region_left
        z_top_bd[:, 1:2] = (region_front - region_behind) * np.random.random([batch_size, 1]) + region_behind
        z_top_bd[:, 2:3] = region_top

    if to_float:
        z_bottom_bd = z_bottom_bd.astype(np.float32)
        z_top_bd = z_top_bd.astype(np.float32)
        x_left_bd = x_left_bd.astype(np.float32)
        x_right_bd = x_right_bd.astype(np.float32)
        y_front_bd = y_front_bd.astype(np.float32)
        y_behind_bd = y_behind_bd.astype(np.float32)
    if to_torch:
        z_bottom_bd = torch.from_numpy(z_bottom_bd)
        z_top_bd = torch.from_numpy(z_top_bd)
        x_left_bd = torch.from_numpy(x_left_bd)
        x_right_bd = torch.from_numpy(x_right_bd)
        y_front_bd = torch.from_numpy(y_front_bd)
        y_behind_bd = torch.from_numpy(y_behind_bd)

        if to_cuda:
            z_bottom_bd = z_bottom_bd.cuda(device='cuda:' + str(gpu_no))
            z_top_bd = z_top_bd.cuda(device='cuda:' + str(gpu_no))
            x_left_bd = x_left_bd.cuda(device='cuda:' + str(gpu_no))
            x_right_bd = x_right_bd.cuda(device='cuda:' + str(gpu_no))
            y_front_bd = y_front_bd.cuda(device='cuda:' + str(gpu_no))
            y_behind_bd = y_behind_bd.cuda(device='cuda:' + str(gpu_no))

        z_bottom_bd.requires_grad = use_grad
        z_top_bd.requires_grad = use_grad
        x_left_bd.requires_grad = use_grad
        x_right_bd.requires_grad = use_grad
        y_front_bd.requires_grad = use_grad
        y_behind_bd.requires_grad = use_grad

    return x_left_bd, x_right_bd, y_front_bd, y_behind_bd, z_bottom_bd, z_top_bd


def rand_bd_4D(batch_size=1000, variable_dim=3, region_xleft=0.0, region_xright=1.0, region_yleft=0.0,
               region_yright=1.0, region_zleft=0.0, region_zright=1.0, region_sleft=0.0, region_sright=1.0,
               to_torch=True, to_float=True, to_cuda=False, gpu_no=0, use_grad=False, opt2sampler='lhs'):
    # np.asarray 将输入转为矩阵格式。
    # 当输入是列表的时候，更改列表的值并不会影响转化为矩阵的值
    # [0,1] 转换为 矩阵，然后
    # reshape(-1,1):数组新的shape属性应该要与原来的配套，如果等于-1的话，那么Numpy会根据剩下的维度计算出数组的另外一个shape属性值。
    assert (int(variable_dim) == 4)
    x00_bd = np.zeros(shape=[batch_size, variable_dim])
    x01_bd = np.zeros(shape=[batch_size, variable_dim])
    y00_bd = np.zeros(shape=[batch_size, variable_dim])
    y01_bd = np.zeros(shape=[batch_size, variable_dim])
    z00_bd = np.zeros(shape=[batch_size, variable_dim])
    z01_bd = np.zeros(shape=[batch_size, variable_dim])
    s00_bd = np.zeros(shape=[batch_size, variable_dim])
    s01_bd = np.zeros(shape=[batch_size, variable_dim])

    if 'lhs' == opt2sampler:
        sampler = stqmc.LatinHypercube(d=1)
        x00_bd[:, 0:1] = region_xleft
        x00_bd[:, 1:2] = (region_yright - region_yleft) * sampler.random(batch_size) + region_yleft
        x00_bd[:, 2:3] = (region_zright - region_zleft) * sampler.random(batch_size) + region_zleft
        x00_bd[:, 3:4] = (region_sright - region_sleft) * sampler.random(batch_size) + region_sleft

        x01_bd[:, 0:1] = region_xright
        x01_bd[:, 1:2] = (region_yright - region_yleft) * sampler.random(batch_size) + region_yleft
        x01_bd[:, 2:3] = (region_zright - region_zleft) * sampler.random(batch_size) + region_zleft
        x01_bd[:, 3:4] = (region_sright - region_sleft) * sampler.random(batch_size) + region_sleft

        y00_bd[:, 0:1] = (region_xright - region_xleft) * sampler.random(batch_size) + region_xleft
        y00_bd[:, 1:2] = region_yleft
        y00_bd[:, 2:3] = (region_zright - region_zleft) * sampler.random(batch_size) + region_zleft
        y00_bd[:, 3:4] = (region_sright - region_sleft) * sampler.random(batch_size) + region_sleft

        y01_bd[:, 0:1] = (region_xright - region_xleft) * sampler.random(batch_size) + region_xleft
        y01_bd[:, 1:2] = region_yright
        y01_bd[:, 2:3] = (region_zright - region_zleft) * sampler.random(batch_size) + region_zleft
        y01_bd[:, 3:4] = (region_sright - region_sleft) * sampler.random(batch_size) + region_sleft

        z00_bd[:, 0:1] = (region_xright - region_xleft) * sampler.random(batch_size) + region_xleft
        z00_bd[:, 1:2] = (region_yright - region_yleft) * sampler.random(batch_size) + region_yleft
        z00_bd[:, 2:3] = region_zleft
        z00_bd[:, 3:4] = (region_sright - region_sleft) * sampler.random(batch_size) + region_sleft

        z01_bd[:, 0:1] = (region_xright - region_xleft) * sampler.random(batch_size) + region_xleft
        z01_bd[:, 1:2] = (region_yright - region_yleft) * sampler.random(batch_size) + region_yleft
        z01_bd[:, 2:3] = region_zright
        z01_bd[:, 3:4] = (region_sright - region_sleft) * sampler.random(batch_size) + region_sleft

        s00_bd[:, 0:1] = (region_xright - region_xleft) * sampler.random(batch_size) + region_xleft
        s00_bd[:, 1:2] = (region_yright - region_yleft) * sampler.random(batch_size) + region_yleft
        s00_bd[:, 2:3] = (region_zright - region_zleft) * sampler.random(batch_size) + region_zleft
        s00_bd[:, 3:4] = region_sleft

        s01_bd[:, 0:1] = (region_xright - region_xleft) * sampler.random(batch_size) + region_xleft
        s01_bd[:, 1:2] = (region_yright - region_yleft) * sampler.random(batch_size) + region_yleft
        s01_bd[:, 2:3] = (region_zright - region_zleft) * sampler.random(batch_size) + region_zleft
        s01_bd[:, 3:4] = region_sright
    else:
        sampler = stqmc.LatinHypercube(d=1)
        x00_bd[:, 0:1] = region_xleft
        x00_bd[:, 1:2] = (region_yright - region_yleft) * np.random.random([batch_size, 1]) + region_yleft
        x00_bd[:, 2:3] = (region_zright - region_zleft) * np.random.random([batch_size, 1]) + region_zleft
        x00_bd[:, 3:4] = (region_sright - region_sleft) * np.random.random([batch_size, 1]) + region_sleft

        x01_bd[:, 0:1] = region_xright
        x01_bd[:, 1:2] = (region_yright - region_yleft) * np.random.random([batch_size, 1]) + region_yleft
        x01_bd[:, 2:3] = (region_zright - region_zleft) * np.random.random([batch_size, 1]) + region_zleft
        x01_bd[:, 3:4] = (region_sright - region_sleft) * np.random.random([batch_size, 1]) + region_sleft

        y00_bd[:, 0:1] = (region_xright - region_xleft) * np.random.random([batch_size, 1]) + region_xleft
        y00_bd[:, 1:2] = region_yleft
        y00_bd[:, 2:3] = (region_zright - region_zleft) * np.random.random([batch_size, 1]) + region_zleft
        y00_bd[:, 3:4] = (region_sright - region_sleft) * np.random.random([batch_size, 1]) + region_sleft

        y01_bd[:, 0:1] = (region_xright - region_xleft) * np.random.random([batch_size, 1]) + region_xleft
        y01_bd[:, 1:2] = region_yright
        y01_bd[:, 2:3] = (region_zright - region_zleft) * np.random.random([batch_size, 1]) + region_zleft
        y01_bd[:, 3:4] = (region_sright - region_sleft) * np.random.random([batch_size, 1]) + region_sleft

        z00_bd[:, 0:1] = (region_xright - region_xleft) * np.random.random([batch_size, 1]) + region_xleft
        z00_bd[:, 1:2] = (region_yright - region_yleft) * np.random.random([batch_size, 1]) + region_yleft
        z00_bd[:, 2:3] = region_zleft
        z00_bd[:, 3:4] = (region_sright - region_sleft) * np.random.random([batch_size, 1]) + region_sleft

        z01_bd[:, 0:1] = (region_xright - region_xleft) * np.random.random([batch_size, 1]) + region_xleft
        z01_bd[:, 1:2] = (region_yright - region_yleft) * np.random.random([batch_size, 1]) + region_yleft
        z01_bd[:, 2:3] = region_zright
        z01_bd[:, 3:4] = (region_sright - region_sleft) * np.random.random([batch_size, 1]) + region_sleft

        s00_bd[:, 0:1] = (region_xright - region_xleft) * np.random.random([batch_size, 1]) + region_xleft
        s00_bd[:, 1:2] = (region_yright - region_yleft) * np.random.random([batch_size, 1]) + region_yleft
        s00_bd[:, 2:3] = (region_zright - region_zleft) * np.random.random([batch_size, 1]) + region_zleft
        s00_bd[:, 3:4] = region_sleft

        s01_bd[:, 0:1] = (region_xright - region_xleft) * np.random.random([batch_size, 1]) + region_xleft
        s01_bd[:, 1:2] = (region_yright - region_yleft) * np.random.random([batch_size, 1]) + region_yleft
        s01_bd[:, 2:3] = (region_zright - region_zleft) * np.random.random([batch_size, 1]) + region_zleft
        s01_bd[:, 3:4] = region_sright

    if to_float:
        x00_bd = x00_bd.astype(np.float32)
        x01_bd = x01_bd.astype(np.float32)

        y00_bd = y00_bd.astype(np.float32)
        y01_bd = y01_bd.astype(np.float32)

        z00_bd = z00_bd.astype(np.float32)
        z01_bd = z01_bd.astype(np.float32)

        s00_bd = s00_bd.astype(np.float32)
        s01_bd = s01_bd.astype(np.float32)

    if to_torch:
        x00_bd = torch.from_numpy(x00_bd)
        x01_bd = torch.from_numpy(x01_bd)

        y00_bd = torch.from_numpy(y00_bd)
        y01_bd = torch.from_numpy(y01_bd)

        z00_bd = torch.from_numpy(z00_bd)
        z01_bd = torch.from_numpy(z01_bd)

        s00_bd = torch.from_numpy(s00_bd)
        s01_bd = torch.from_numpy(s01_bd)

        if to_cuda:
            x00_bd = x00_bd.cuda(device='cuda:' + str(gpu_no))
            x01_bd = x01_bd.cuda(device='cuda:' + str(gpu_no))

            y00_bd = y00_bd.cuda(device='cuda:' + str(gpu_no))
            y01_bd = y01_bd.cuda(device='cuda:' + str(gpu_no))

            z00_bd = z00_bd.cuda(device='cuda:' + str(gpu_no))
            z01_bd = z01_bd.cuda(device='cuda:' + str(gpu_no))

            s00_bd = s00_bd.cuda(device='cuda:' + str(gpu_no))
            s01_bd = s01_bd.cuda(device='cuda:' + str(gpu_no))

        x00_bd.requires_grad = use_grad
        x01_bd.requires_grad = use_grad
        y00_bd.requires_grad = use_grad
        y01_bd.requires_grad = use_grad
        z00_bd.requires_grad = use_grad
        z01_bd.requires_grad = use_grad
        s00_bd.requires_grad = use_grad
        s01_bd.requires_grad = use_grad

    return x00_bd, x01_bd, y00_bd, y01_bd, z00_bd, z01_bd, s00_bd, s01_bd


def rand_bd_5D(batch_size=1000, variable_dim=5, region_xleft=0.0, region_xright=1.0, region_yleft=0.0, region_yright=1.0,
               region_zleft=0.0, region_zright=1.0, region_sleft=0.0, region_sright=1.0, region_tleft=0.0,
               region_tright=1.0, to_torch=True, to_float=True, to_cuda=False, gpu_no=0, use_grad=False,
               opt2sampler='lhs'):
    # np.asarray 将输入转为矩阵格式。
    # 当输入是列表的时候，更改列表的值并不会影响转化为矩阵的值
    # [0,1] 转换为 矩阵，然后
    # reshape(-1,1):数组新的shape属性应该要与原来的配套，如果等于-1的话，那么Numpy会根据剩下的维度计算出数组的另外一个shape属性值。
    assert variable_dim == 5
    x00_bd = np.zeros(shape=[batch_size, variable_dim])
    x01_bd = np.zeros(shape=[batch_size, variable_dim])

    y00_bd = np.zeros(shape=[batch_size, variable_dim])
    y01_bd = np.zeros(shape=[batch_size, variable_dim])

    z00_bd = np.zeros(shape=[batch_size, variable_dim])
    z01_bd = np.zeros(shape=[batch_size, variable_dim])

    s00_bd = np.zeros(shape=[batch_size, variable_dim])
    s01_bd = np.zeros(shape=[batch_size, variable_dim])

    t00_bd = np.zeros(shape=[batch_size, variable_dim])
    t01_bd = np.zeros(shape=[batch_size, variable_dim])

    if 'lhs' == opt2sampler:
        sampler = stqmc.LatinHypercube(d=1)
        x00_bd[:, 0:1] = region_xleft
        x00_bd[:, 1:2] = (region_yright - region_yleft) * sampler.random(batch_size) + region_yleft
        x00_bd[:, 2:3] = (region_zright - region_zleft) * sampler.random(batch_size) + region_zleft
        x00_bd[:, 3:4] = (region_sright - region_sleft) * sampler.random(batch_size) + region_sleft
        x00_bd[:, 4:5] = (region_tright - region_tleft) * sampler.random(batch_size) + region_tleft

        x01_bd[:, 0:1] = region_xright
        x01_bd[:, 1:2] = (region_yright - region_yleft) * sampler.random(batch_size) + region_yleft
        x01_bd[:, 2:3] = (region_zright - region_zleft) * sampler.random(batch_size) + region_zleft
        x01_bd[:, 3:4] = (region_sright - region_sleft) * sampler.random(batch_size) + region_sleft
        x01_bd[:, 4:5] = (region_tright - region_tleft) * sampler.random(batch_size) + region_tleft

        y00_bd[:, 0:1] = (region_xright - region_xleft) * sampler.random(batch_size) + region_xleft
        y00_bd[:, 1:2] = region_yleft
        y00_bd[:, 2:3] = (region_zright - region_zleft) * sampler.random(batch_size) + region_zleft
        y00_bd[:, 3:4] = (region_sright - region_sleft) * sampler.random(batch_size) + region_sleft
        y00_bd[:, 4:5] = (region_tright - region_tleft) * sampler.random(batch_size) + region_tleft

        y01_bd[:, 0:1] = (region_xright - region_xleft) * sampler.random(batch_size) + region_xleft
        y01_bd[:, 1:2] = region_yright
        y01_bd[:, 2:3] = (region_zright - region_zleft) * sampler.random(batch_size) + region_zleft
        y01_bd[:, 3:4] = (region_sright - region_sleft) * sampler.random(batch_size) + region_sleft
        y01_bd[:, 4:5] = (region_tright - region_tleft) * sampler.random(batch_size) + region_tleft

        z00_bd[:, 0:1] = (region_xright - region_xleft) * sampler.random(batch_size) + region_xleft
        z00_bd[:, 1:2] = (region_yright - region_yleft) * sampler.random(batch_size) + region_yleft
        z00_bd[:, 2:3] = region_zleft
        z00_bd[:, 3:4] = (region_sright - region_sleft) * sampler.random(batch_size) + region_sleft
        z00_bd[:, 4:5] = (region_tright - region_tleft) * sampler.random(batch_size) + region_tleft

        z01_bd[:, 0:1] = (region_xright - region_xleft) * sampler.random(batch_size) + region_xleft
        z01_bd[:, 1:2] = (region_yright - region_yleft) * sampler.random(batch_size) + region_yleft
        z01_bd[:, 2:3] = region_zright
        z01_bd[:, 3:4] = (region_sright - region_sleft) * sampler.random(batch_size) + region_sleft
        z01_bd[:, 4:5] = (region_tright - region_tleft) * sampler.random(batch_size) + region_tleft

        s00_bd[:, 0:1] = (region_xright - region_xleft) * sampler.random(batch_size) + region_xleft
        s00_bd[:, 1:2] = (region_yright - region_yleft) * sampler.random(batch_size) + region_yleft
        s00_bd[:, 2:3] = (region_zright - region_zleft) * sampler.random(batch_size) + region_zleft
        s00_bd[:, 3:4] = region_sleft
        s00_bd[:, 4:5] = (region_tright - region_tleft) * sampler.random(batch_size) + region_tleft

        s01_bd[:, 0:1] = (region_xright - region_xleft) * sampler.random(batch_size) + region_xleft
        s01_bd[:, 1:2] = (region_yright - region_yleft) * sampler.random(batch_size) + region_yleft
        s01_bd[:, 2:3] = (region_zright - region_zleft) * sampler.random(batch_size) + region_zleft
        s01_bd[:, 3:4] = region_sright
        s01_bd[:, 4:5] = (region_tright - region_tleft) * sampler.random(batch_size) + region_tleft

        t00_bd[:, 0:1] = (region_xright - region_xleft) * sampler.random(batch_size) + region_xleft
        t00_bd[:, 1:2] = (region_yright - region_yleft) * sampler.random(batch_size) + region_yleft
        t00_bd[:, 2:3] = (region_zright - region_zleft) * sampler.random(batch_size) + region_zleft
        t00_bd[:, 3:4] = (region_sright - region_sleft) * sampler.random(batch_size) + region_sleft
        t00_bd[:, 4:5] = region_tleft

        t01_bd[:, 0:1] = (region_xright - region_xleft) * sampler.random(batch_size) + region_xleft
        t01_bd[:, 1:2] = (region_yright - region_yleft) * sampler.random(batch_size) + region_yleft
        t01_bd[:, 2:3] = (region_zright - region_zleft) * sampler.random(batch_size) + region_zleft
        t01_bd[:, 3:4] = (region_sright - region_sleft) * sampler.random(batch_size) + region_sleft
        t01_bd[:, 4:5] = region_tright
    else:
        x00_bd[:, 0:1] = region_xleft
        x00_bd[:, 1:2] = (region_yright - region_yleft) * np.random.random([batch_size, 1]) + region_yleft
        x00_bd[:, 2:3] = (region_zright - region_zleft) * np.random.random([batch_size, 1]) + region_zleft
        x00_bd[:, 3:4] = (region_sright - region_sleft) * np.random.random([batch_size, 1]) + region_sleft
        x00_bd[:, 4:5] = (region_tright - region_tleft) * np.random.random([batch_size, 1]) + region_tleft

        x01_bd[:, 0:1] = region_xright
        x01_bd[:, 1:2] = (region_yright - region_yleft) * np.random.random([batch_size, 1]) + region_yleft
        x01_bd[:, 2:3] = (region_zright - region_zleft) * np.random.random([batch_size, 1]) + region_zleft
        x01_bd[:, 3:4] = (region_sright - region_sleft) * np.random.random([batch_size, 1]) + region_sleft
        x01_bd[:, 4:5] = (region_tright - region_tleft) * np.random.random([batch_size, 1]) + region_tleft

        y00_bd[:, 0:1] = (region_xright - region_xleft) * np.random.random([batch_size, 1]) + region_xleft
        y00_bd[:, 1:2] = region_yleft
        y00_bd[:, 2:3] = (region_zright - region_zleft) * np.random.random([batch_size, 1]) + region_zleft
        y00_bd[:, 3:4] = (region_sright - region_sleft) * np.random.random([batch_size, 1]) + region_sleft
        y00_bd[:, 4:5] = (region_tright - region_tleft) * np.random.random([batch_size, 1]) + region_tleft

        y01_bd[:, 0:1] = (region_xright - region_xleft) * np.random.random([batch_size, 1]) + region_xleft
        y01_bd[:, 1:2] = region_yright
        y01_bd[:, 2:3] = (region_zright - region_zleft) * np.random.random([batch_size, 1]) + region_zleft
        y01_bd[:, 3:4] = (region_sright - region_sleft) * np.random.random([batch_size, 1]) + region_sleft
        y01_bd[:, 4:5] = (region_tright - region_tleft) * np.random.random([batch_size, 1]) + region_tleft

        z00_bd[:, 0:1] = (region_xright - region_xleft) * np.random.random([batch_size, 1]) + region_xleft
        z00_bd[:, 1:2] = (region_yright - region_yleft) * np.random.random([batch_size, 1]) + region_yleft
        z00_bd[:, 2:3] = region_zleft
        z00_bd[:, 3:4] = (region_sright - region_sleft) * np.random.random([batch_size, 1]) + region_sleft
        z00_bd[:, 4:5] = (region_tright - region_tleft) * np.random.random([batch_size, 1]) + region_tleft

        z01_bd[:, 0:1] = (region_xright - region_xleft) * np.random.random([batch_size, 1]) + region_xleft
        z01_bd[:, 1:2] = (region_yright - region_yleft) * np.random.random([batch_size, 1]) + region_yleft
        z01_bd[:, 2:3] = region_zright
        z01_bd[:, 3:4] = (region_sright - region_sleft) * np.random.random([batch_size, 1]) + region_sleft
        z01_bd[:, 4:5] = (region_tright - region_tleft) * np.random.random([batch_size, 1]) + region_tleft

        s00_bd[:, 0:1] = (region_xright - region_xleft) * np.random.random([batch_size, 1]) + region_xleft
        s00_bd[:, 1:2] = (region_yright - region_yleft) * np.random.random([batch_size, 1]) + region_yleft
        s00_bd[:, 2:3] = (region_zright - region_zleft) * np.random.random([batch_size, 1]) + region_zleft
        s00_bd[:, 3:4] = region_sleft
        s00_bd[:, 4:5] = (region_tright - region_tleft) * np.random.random([batch_size, 1]) + region_tleft

        s01_bd[:, 0:1] = (region_xright - region_xleft) * np.random.random([batch_size, 1]) + region_xleft
        s01_bd[:, 1:2] = (region_yright - region_yleft) * np.random.random([batch_size, 1]) + region_yleft
        s01_bd[:, 2:3] = (region_zright - region_zleft) * np.random.random([batch_size, 1]) + region_zleft
        s01_bd[:, 3:4] = region_sright
        s01_bd[:, 4:5] = (region_tright - region_tleft) * np.random.random([batch_size, 1]) + region_tleft

        t00_bd[:, 0:1] = (region_xright - region_xleft) * np.random.random([batch_size, 1]) + region_xleft
        t00_bd[:, 1:2] = (region_yright - region_yleft) * np.random.random([batch_size, 1]) + region_yleft
        t00_bd[:, 2:3] = (region_zright - region_zleft) * np.random.random([batch_size, 1]) + region_zleft
        t00_bd[:, 3:4] = (region_sright - region_sleft) * np.random.random([batch_size, 1]) + region_sleft
        t00_bd[:, 4:5] = region_tleft

        t01_bd[:, 0:1] = (region_xright - region_xleft) * np.random.random([batch_size, 1]) + region_xleft
        t01_bd[:, 1:2] = (region_yright - region_yleft) * np.random.random([batch_size, 1]) + region_yleft
        t01_bd[:, 2:3] = (region_zright - region_zleft) * np.random.random([batch_size, 1]) + region_zleft
        t01_bd[:, 3:4] = (region_sright - region_sleft) * np.random.random([batch_size, 1]) + region_sleft
        t01_bd[:, 4:5] = region_tright

    if to_float:
        x00_bd = x00_bd.astype(np.float32)
        x01_bd = x01_bd.astype(np.float32)

        y00_bd = y00_bd.astype(np.float32)
        y01_bd = y01_bd.astype(np.float32)

        z00_bd = z00_bd.astype(np.float32)
        z01_bd = z01_bd.astype(np.float32)

        s00_bd = s00_bd.astype(np.float32)
        s01_bd = s01_bd.astype(np.float32)

        t00_bd = t00_bd.astype(np.float32)
        t01_bd = t01_bd.astype(np.float32)

    if to_torch:
        x00_bd = torch.from_numpy(x00_bd)
        x01_bd = torch.from_numpy(x01_bd)

        y00_bd = torch.from_numpy(y00_bd)
        y01_bd = torch.from_numpy(y01_bd)

        z00_bd = torch.from_numpy(z00_bd)
        z01_bd = torch.from_numpy(z01_bd)

        s00_bd = torch.from_numpy(s00_bd)
        s01_bd = torch.from_numpy(s01_bd)

        t00_bd = torch.from_numpy(t00_bd)
        t01_bd = torch.from_numpy(t01_bd)

        if to_cuda:
            x00_bd = x00_bd.cuda(device='cuda:' + str(gpu_no))
            x01_bd = x01_bd.cuda(device='cuda:' + str(gpu_no))

            y00_bd = y00_bd.cuda(device='cuda:' + str(gpu_no))
            y01_bd = y01_bd.cuda(device='cuda:' + str(gpu_no))

            z00_bd = z00_bd.cuda(device='cuda:' + str(gpu_no))
            z01_bd = z01_bd.cuda(device='cuda:' + str(gpu_no))

            s00_bd = s00_bd.cuda(device='cuda:' + str(gpu_no))
            s01_bd = s01_bd.cuda(device='cuda:' + str(gpu_no))

            t00_bd = t00_bd.cuda(device='cuda:' + str(gpu_no))
            t01_bd = t01_bd.cuda(device='cuda:' + str(gpu_no))

        x00_bd.requires_grad = use_grad
        x01_bd.requires_grad = use_grad
        y00_bd.requires_grad = use_grad
        y01_bd.requires_grad = use_grad
        z00_bd.requires_grad = use_grad
        z01_bd.requires_grad = use_grad
        s00_bd.requires_grad = use_grad
        s01_bd.requires_grad = use_grad
        t00_bd.requires_grad = use_grad
        t01_bd.requires_grad = use_grad

    return x00_bd, x01_bd, y00_bd, y01_bd, z00_bd, z01_bd, s00_bd, s01_bd, t00_bd, t01_bd


def rand_bd_8D(batch_size=1000, variable_dim=8, region_xleft=0.0, region_xright=1.0, region_yleft=0.0, region_yright=1.0,
               region_zleft=0.0, region_zright=1.0, region_sleft=0.0, region_sright=1.0, region_tleft=0.0,
               region_tright=1.0, region_pleft=0.0, region_pright=1.0, region_qleft=0.0, region_qright=1.0,
               region_rleft=0.0, region_rright=1.0, to_torch=True, to_float=True, to_cuda=False, gpu_no=0,
               use_grad=False, opt2sampler='lhs'):
    # np.asarray 将输入转为矩阵格式。
    # 当输入是列表的时候，更改列表的值并不会影响转化为矩阵的值
    # [0,1] 转换为 矩阵，然后
    # reshape(-1,1):数组新的shape属性应该要与原来的配套，如果等于-1的话，那么Numpy会根据剩下的维度计算出数组的另外一个shape属性值。
    assert variable_dim == 8
    x00_bd = np.zeros(shape=[batch_size, variable_dim])
    x01_bd = np.zeros(shape=[batch_size, variable_dim])

    y00_bd = np.zeros(shape=[batch_size, variable_dim])
    y01_bd = np.zeros(shape=[batch_size, variable_dim])

    z00_bd = np.zeros(shape=[batch_size, variable_dim])
    z01_bd = np.zeros(shape=[batch_size, variable_dim])

    s00_bd = np.zeros(shape=[batch_size, variable_dim])
    s01_bd = np.zeros(shape=[batch_size, variable_dim])

    t00_bd = np.zeros(shape=[batch_size, variable_dim])
    t01_bd = np.zeros(shape=[batch_size, variable_dim])

    p00_bd = np.zeros(shape=[batch_size, variable_dim])
    p01_bd = np.zeros(shape=[batch_size, variable_dim])

    q00_bd = np.zeros(shape=[batch_size, variable_dim])
    q01_bd = np.zeros(shape=[batch_size, variable_dim])

    r00_bd = np.zeros(shape=[batch_size, variable_dim])
    r01_bd = np.zeros(shape=[batch_size, variable_dim])

    if 'lhs' == opt2sampler:
        sampler = stqmc.LatinHypercube(d=1)
        x00_bd[:, 0:1] = region_xleft
        x00_bd[:, 1:2] = (region_yright - region_yleft) * sampler.random(batch_size) + region_yleft
        x00_bd[:, 2:3] = (region_zright - region_zleft) * sampler.random(batch_size) + region_zleft
        x00_bd[:, 3:4] = (region_sright - region_sleft) * sampler.random(batch_size) + region_sleft
        x00_bd[:, 4:5] = (region_tright - region_tleft) * sampler.random(batch_size) + region_tleft
        x00_bd[:, 5:6] = (region_pright - region_pleft) * sampler.random(batch_size) + region_pleft
        x00_bd[:, 6:7] = (region_qright - region_qleft) * sampler.random(batch_size) + region_qleft
        x00_bd[:, 7:8] = (region_rright - region_rleft) * sampler.random(batch_size) + region_rleft

        x01_bd[:, 0:1] = region_xright
        x01_bd[:, 1:2] = (region_yright - region_yleft) * sampler.random(batch_size) + region_yleft
        x01_bd[:, 2:3] = (region_zright - region_zleft) * sampler.random(batch_size) + region_zleft
        x01_bd[:, 3:4] = (region_sright - region_sleft) * sampler.random(batch_size) + region_sleft
        x01_bd[:, 4:5] = (region_tright - region_tleft) * sampler.random(batch_size) + region_tleft
        x01_bd[:, 5:6] = (region_pright - region_pleft) * sampler.random(batch_size) + region_pleft
        x01_bd[:, 6:7] = (region_qright - region_qleft) * sampler.random(batch_size) + region_qleft
        x01_bd[:, 7:8] = (region_rright - region_rleft) * sampler.random(batch_size) + region_rleft

        y00_bd[:, 0:1] = (region_xright - region_xleft) * sampler.random(batch_size) + region_xleft
        y00_bd[:, 1:2] = region_yleft
        y00_bd[:, 2:3] = (region_zright - region_zleft) * sampler.random(batch_size) + region_zleft
        y00_bd[:, 3:4] = (region_sright - region_sleft) * sampler.random(batch_size) + region_sleft
        y00_bd[:, 4:5] = (region_tright - region_tleft) * sampler.random(batch_size) + region_tleft
        y00_bd[:, 5:6] = (region_pright - region_pleft) * sampler.random(batch_size) + region_pleft
        y00_bd[:, 6:7] = (region_qright - region_qleft) * sampler.random(batch_size) + region_qleft
        y00_bd[:, 7:8] = (region_rright - region_rleft) * sampler.random(batch_size) + region_rleft

        y01_bd[:, 0:1] = (region_xright - region_xleft) * sampler.random(batch_size) + region_xleft
        y01_bd[:, 1:2] = region_yright
        y01_bd[:, 2:3] = (region_zright - region_zleft) * sampler.random(batch_size) + region_zleft
        y01_bd[:, 3:4] = (region_sright - region_sleft) * sampler.random(batch_size) + region_sleft
        y01_bd[:, 4:5] = (region_tright - region_tleft) * sampler.random(batch_size) + region_tleft
        y01_bd[:, 5:6] = (region_pright - region_pleft) * sampler.random(batch_size) + region_pleft
        y01_bd[:, 6:7] = (region_qright - region_qleft) * sampler.random(batch_size) + region_qleft
        y01_bd[:, 7:8] = (region_rright - region_rleft) * sampler.random(batch_size) + region_rleft

        z00_bd[:, 0:1] = (region_xright - region_xleft) * sampler.random(batch_size) + region_xleft
        z00_bd[:, 1:2] = (region_yright - region_yleft) * sampler.random(batch_size) + region_yleft
        z00_bd[:, 2:3] = region_zleft
        z00_bd[:, 3:4] = (region_sright - region_sleft) * sampler.random(batch_size) + region_sleft
        z00_bd[:, 4:5] = (region_tright - region_tleft) * sampler.random(batch_size) + region_tleft
        z00_bd[:, 5:6] = (region_pright - region_pleft) * sampler.random(batch_size) + region_pleft
        z00_bd[:, 6:7] = (region_qright - region_qleft) * sampler.random(batch_size) + region_qleft
        z00_bd[:, 7:8] = (region_rright - region_rleft) * sampler.random(batch_size) + region_rleft

        z01_bd[:, 0:1] = (region_xright - region_xleft) * sampler.random(batch_size) + region_xleft
        z01_bd[:, 1:2] = (region_yright - region_yleft) * sampler.random(batch_size) + region_yleft
        z01_bd[:, 2:3] = region_zright
        z01_bd[:, 3:4] = (region_sright - region_sleft) * sampler.random(batch_size) + region_sleft
        z01_bd[:, 4:5] = (region_tright - region_tleft) * sampler.random(batch_size) + region_tleft
        z01_bd[:, 5:6] = (region_pright - region_pleft) * sampler.random(batch_size) + region_pleft
        z01_bd[:, 6:7] = (region_qright - region_qleft) * sampler.random(batch_size) + region_qleft
        z01_bd[:, 7:8] = (region_rright - region_rleft) * sampler.random(batch_size) + region_rleft

        s00_bd[:, 0:1] = (region_xright - region_xleft) * sampler.random(batch_size) + region_xleft
        s00_bd[:, 1:2] = (region_yright - region_yleft) * sampler.random(batch_size) + region_yleft
        s00_bd[:, 2:3] = (region_zright - region_zleft) * sampler.random(batch_size) + region_zleft
        s00_bd[:, 3:4] = region_sleft
        s00_bd[:, 4:5] = (region_tright - region_tleft) * sampler.random(batch_size) + region_tleft
        s00_bd[:, 5:6] = (region_pright - region_pleft) * sampler.random(batch_size) + region_pleft
        s00_bd[:, 6:7] = (region_qright - region_qleft) * sampler.random(batch_size) + region_qleft
        s00_bd[:, 7:8] = (region_rright - region_rleft) * sampler.random(batch_size) + region_rleft

        s01_bd[:, 0:1] = (region_xright - region_xleft) * sampler.random(batch_size) + region_xleft
        s01_bd[:, 1:2] = (region_yright - region_yleft) * sampler.random(batch_size) + region_yleft
        s01_bd[:, 2:3] = (region_zright - region_zleft) * sampler.random(batch_size) + region_zleft
        s01_bd[:, 3:4] = region_sright
        s01_bd[:, 4:5] = (region_tright - region_tleft) * sampler.random(batch_size) + region_tleft
        z01_bd[:, 5:6] = (region_pright - region_pleft) * sampler.random(batch_size) + region_pleft
        z01_bd[:, 6:7] = (region_qright - region_qleft) * sampler.random(batch_size) + region_qleft
        z01_bd[:, 7:8] = (region_rright - region_rleft) * sampler.random(batch_size) + region_rleft

        t00_bd[:, 0:1] = (region_xright - region_xleft) * sampler.random(batch_size) + region_xleft
        t00_bd[:, 1:2] = (region_yright - region_yleft) * sampler.random(batch_size) + region_yleft
        t00_bd[:, 2:3] = (region_zright - region_zleft) * sampler.random(batch_size) + region_zleft
        t00_bd[:, 3:4] = (region_sright - region_sleft) * sampler.random(batch_size) + region_sleft
        t00_bd[:, 4:5] = region_tleft
        t00_bd[:, 5:6] = (region_pright - region_pleft) * sampler.random(batch_size) + region_pleft
        t00_bd[:, 6:7] = (region_qright - region_qleft) * sampler.random(batch_size) + region_qleft
        t00_bd[:, 7:8] = (region_rright - region_rleft) * sampler.random(batch_size) + region_rleft

        t01_bd[:, 0:1] = (region_xright - region_xleft) * sampler.random(batch_size) + region_xleft
        t01_bd[:, 1:2] = (region_yright - region_yleft) * sampler.random(batch_size) + region_yleft
        t01_bd[:, 2:3] = (region_zright - region_zleft) * sampler.random(batch_size) + region_zleft
        t01_bd[:, 3:4] = (region_sright - region_sleft) * sampler.random(batch_size) + region_sleft
        t01_bd[:, 4:5] = region_tright
        t01_bd[:, 5:6] = (region_pright - region_pleft) * sampler.random(batch_size) + region_pleft
        t01_bd[:, 6:7] = (region_qright - region_qleft) * sampler.random(batch_size) + region_qleft
        t01_bd[:, 7:8] = (region_rright - region_rleft) * sampler.random(batch_size) + region_rleft

        p00_bd[:, 0:1] = (region_xright - region_xleft) * sampler.random(batch_size) + region_xleft
        p00_bd[:, 1:2] = (region_yright - region_yleft) * sampler.random(batch_size) + region_yleft
        p00_bd[:, 2:3] = (region_zright - region_zleft) * sampler.random(batch_size) + region_zleft
        p00_bd[:, 3:4] = (region_sright - region_sleft) * sampler.random(batch_size) + region_sleft
        p00_bd[:, 4:5] = (region_tright - region_tleft) * sampler.random(batch_size) + region_tleft
        p00_bd[:, 5:6] = region_pleft
        p00_bd[:, 6:7] = (region_qright - region_qleft) * sampler.random(batch_size) + region_qleft
        p00_bd[:, 7:8] = (region_rright - region_rleft) * sampler.random(batch_size) + region_rleft

        p01_bd[:, 0:1] = (region_xright - region_xleft) * sampler.random(batch_size) + region_xleft
        p01_bd[:, 1:2] = (region_yright - region_yleft) * sampler.random(batch_size) + region_yleft
        p01_bd[:, 2:3] = (region_zright - region_zleft) * sampler.random(batch_size) + region_zleft
        p01_bd[:, 3:4] = (region_sright - region_sleft) * sampler.random(batch_size) + region_sleft
        p01_bd[:, 4:5] = (region_tright - region_tleft) * sampler.random(batch_size) + region_tleft
        p01_bd[:, 5:6] = region_pright
        p01_bd[:, 6:7] = (region_qright - region_qleft) * sampler.random(batch_size) + region_qleft
        p01_bd[:, 7:8] = (region_rright - region_rleft) * sampler.random(batch_size) + region_rleft

        q00_bd[:, 0:1] = (region_xright - region_xleft) * sampler.random(batch_size) + region_xleft
        q00_bd[:, 1:2] = (region_yright - region_yleft) * sampler.random(batch_size) + region_yleft
        q00_bd[:, 2:3] = (region_zright - region_zleft) * sampler.random(batch_size) + region_zleft
        q00_bd[:, 3:4] = (region_sright - region_sleft) * sampler.random(batch_size) + region_sleft
        q00_bd[:, 4:5] = (region_tright - region_tleft) * sampler.random(batch_size) + region_tleft
        q00_bd[:, 5:6] = (region_pright - region_pleft) * sampler.random(batch_size) + region_pleft
        q00_bd[:, 6:7] = region_qleft
        q00_bd[:, 7:8] = (region_rright - region_rleft) * sampler.random(batch_size) + region_rleft

        q01_bd[:, 0:1] = (region_xright - region_xleft) * sampler.random(batch_size) + region_xleft
        q01_bd[:, 1:2] = (region_yright - region_yleft) * sampler.random(batch_size) + region_yleft
        q01_bd[:, 2:3] = (region_zright - region_zleft) * sampler.random(batch_size) + region_zleft
        q01_bd[:, 3:4] = (region_sright - region_sleft) * sampler.random(batch_size) + region_sleft
        q01_bd[:, 4:5] = (region_tright - region_tleft) * sampler.random(batch_size) + region_tleft
        q01_bd[:, 5:6] = (region_pright - region_pleft) * sampler.random(batch_size) + region_pleft
        q01_bd[:, 6:7] = region_qright
        q01_bd[:, 7:8] = (region_rright - region_rleft) * sampler.random(batch_size) + region_rleft

        r00_bd[:, 0:1] = (region_xright - region_xleft) * sampler.random(batch_size) + region_xleft
        r00_bd[:, 1:2] = (region_yright - region_yleft) * sampler.random(batch_size) + region_yleft
        r00_bd[:, 2:3] = (region_zright - region_zleft) * sampler.random(batch_size) + region_zleft
        r00_bd[:, 3:4] = (region_sright - region_sleft) * sampler.random(batch_size) + region_sleft
        r00_bd[:, 4:5] = (region_tright - region_tleft) * sampler.random(batch_size) + region_tleft
        r00_bd[:, 5:6] = (region_pright - region_pleft) * sampler.random(batch_size) + region_pleft
        r00_bd[:, 6:7] = (region_qright - region_qleft) * sampler.random(batch_size) + region_qleft
        r00_bd[:, 7:8] = region_rleft

        r01_bd[:, 0:1] = (region_xright - region_xleft) * sampler.random(batch_size) + region_xleft
        r01_bd[:, 1:2] = (region_yright - region_yleft) * sampler.random(batch_size) + region_yleft
        r01_bd[:, 2:3] = (region_zright - region_zleft) * sampler.random(batch_size) + region_zleft
        r01_bd[:, 3:4] = (region_sright - region_sleft) * sampler.random(batch_size) + region_sleft
        r01_bd[:, 4:5] = (region_tright - region_tleft) * sampler.random(batch_size) + region_tleft
        r01_bd[:, 5:6] = (region_pright - region_pleft) * sampler.random(batch_size) + region_pleft
        r01_bd[:, 6:7] = (region_qright - region_qleft) * sampler.random(batch_size) + region_qleft
        r01_bd[:, 7:8] = region_rright
    else:
        x00_bd[:, 0:1] = region_xleft
        x00_bd[:, 1:2] = (region_yright - region_yleft) * np.random.random([batch_size, 1]) + region_yleft
        x00_bd[:, 2:3] = (region_zright - region_zleft) * np.random.random([batch_size, 1]) + region_zleft
        x00_bd[:, 3:4] = (region_sright - region_sleft) * np.random.random([batch_size, 1]) + region_sleft
        x00_bd[:, 4:5] = (region_tright - region_tleft) * np.random.random([batch_size, 1]) + region_tleft
        x00_bd[:, 5:6] = (region_pright - region_pleft) * np.random.random([batch_size, 1]) + region_pleft
        x00_bd[:, 6:7] = (region_qright - region_qleft) * np.random.random([batch_size, 1]) + region_qleft
        x00_bd[:, 7:8] = (region_rright - region_rleft) * np.random.random([batch_size, 1]) + region_rleft

        x01_bd[:, 0:1] = region_xright
        x01_bd[:, 1:2] = (region_yright - region_yleft) * np.random.random([batch_size, 1]) + region_yleft
        x01_bd[:, 2:3] = (region_zright - region_zleft) * np.random.random([batch_size, 1]) + region_zleft
        x01_bd[:, 3:4] = (region_sright - region_sleft) * np.random.random([batch_size, 1]) + region_sleft
        x01_bd[:, 4:5] = (region_tright - region_tleft) * np.random.random([batch_size, 1]) + region_tleft
        x01_bd[:, 5:6] = (region_pright - region_pleft) * np.random.random([batch_size, 1]) + region_pleft
        x01_bd[:, 6:7] = (region_qright - region_qleft) * np.random.random([batch_size, 1]) + region_qleft
        x01_bd[:, 7:8] = (region_rright - region_rleft) * np.random.random([batch_size, 1]) + region_rleft

        y00_bd[:, 0:1] = (region_xright - region_xleft) * np.random.random([batch_size, 1]) + region_xleft
        y00_bd[:, 1:2] = region_yleft
        y00_bd[:, 2:3] = (region_zright - region_zleft) * np.random.random([batch_size, 1]) + region_zleft
        y00_bd[:, 3:4] = (region_sright - region_sleft) * np.random.random([batch_size, 1]) + region_sleft
        y00_bd[:, 4:5] = (region_tright - region_tleft) * np.random.random([batch_size, 1]) + region_tleft
        y00_bd[:, 5:6] = (region_pright - region_pleft) * np.random.random([batch_size, 1]) + region_pleft
        y00_bd[:, 6:7] = (region_qright - region_qleft) * np.random.random([batch_size, 1]) + region_qleft
        y00_bd[:, 7:8] = (region_rright - region_rleft) * np.random.random([batch_size, 1]) + region_rleft

        y01_bd[:, 0:1] = (region_xright - region_xleft) * np.random.random([batch_size, 1]) + region_xleft
        y01_bd[:, 1:2] = region_yright
        y01_bd[:, 2:3] = (region_zright - region_zleft) * np.random.random([batch_size, 1]) + region_zleft
        y01_bd[:, 3:4] = (region_sright - region_sleft) * np.random.random([batch_size, 1]) + region_sleft
        y01_bd[:, 4:5] = (region_tright - region_tleft) * np.random.random([batch_size, 1]) + region_tleft
        y01_bd[:, 5:6] = (region_pright - region_pleft) * np.random.random([batch_size, 1]) + region_pleft
        y01_bd[:, 6:7] = (region_qright - region_qleft) * np.random.random([batch_size, 1]) + region_qleft
        y01_bd[:, 7:8] = (region_rright - region_rleft) * np.random.random([batch_size, 1]) + region_rleft

        z00_bd[:, 0:1] = (region_xright - region_xleft) * np.random.random([batch_size, 1]) + region_xleft
        z00_bd[:, 1:2] = (region_yright - region_yleft) * np.random.random([batch_size, 1]) + region_yleft
        z00_bd[:, 2:3] = region_zleft
        z00_bd[:, 3:4] = (region_sright - region_sleft) * np.random.random([batch_size, 1]) + region_sleft
        z00_bd[:, 4:5] = (region_tright - region_tleft) * np.random.random([batch_size, 1]) + region_tleft
        z00_bd[:, 5:6] = (region_pright - region_pleft) * np.random.random([batch_size, 1]) + region_pleft
        z00_bd[:, 6:7] = (region_qright - region_qleft) * np.random.random([batch_size, 1]) + region_qleft
        z00_bd[:, 7:8] = (region_rright - region_rleft) * np.random.random([batch_size, 1]) + region_rleft

        z01_bd[:, 0:1] = (region_xright - region_xleft) * np.random.random([batch_size, 1]) + region_xleft
        z01_bd[:, 1:2] = (region_yright - region_yleft) * np.random.random([batch_size, 1]) + region_yleft
        z01_bd[:, 2:3] = region_zright
        z01_bd[:, 3:4] = (region_sright - region_sleft) * np.random.random([batch_size, 1]) + region_sleft
        z01_bd[:, 4:5] = (region_tright - region_tleft) * np.random.random([batch_size, 1]) + region_tleft
        z01_bd[:, 5:6] = (region_pright - region_pleft) * np.random.random([batch_size, 1]) + region_pleft
        z01_bd[:, 6:7] = (region_qright - region_qleft) * np.random.random([batch_size, 1]) + region_qleft
        z01_bd[:, 7:8] = (region_rright - region_rleft) * np.random.random([batch_size, 1]) + region_rleft

        s00_bd[:, 0:1] = (region_xright - region_xleft) * np.random.random([batch_size, 1]) + region_xleft
        s00_bd[:, 1:2] = (region_yright - region_yleft) * np.random.random([batch_size, 1]) + region_yleft
        s00_bd[:, 2:3] = (region_zright - region_zleft) * np.random.random([batch_size, 1]) + region_zleft
        s00_bd[:, 3:4] = region_sleft
        s00_bd[:, 4:5] = (region_tright - region_tleft) * np.random.random([batch_size, 1]) + region_tleft
        s00_bd[:, 5:6] = (region_pright - region_pleft) * np.random.random([batch_size, 1]) + region_pleft
        s00_bd[:, 6:7] = (region_qright - region_qleft) * np.random.random([batch_size, 1]) + region_qleft
        s00_bd[:, 7:8] = (region_rright - region_rleft) * np.random.random([batch_size, 1]) + region_rleft

        s01_bd[:, 0:1] = (region_xright - region_xleft) * np.random.random([batch_size, 1]) + region_xleft
        s01_bd[:, 1:2] = (region_yright - region_yleft) * np.random.random([batch_size, 1]) + region_yleft
        s01_bd[:, 2:3] = (region_zright - region_zleft) * np.random.random([batch_size, 1]) + region_zleft
        s01_bd[:, 3:4] = region_sright
        s01_bd[:, 4:5] = (region_tright - region_tleft) * np.random.random([batch_size, 1]) + region_tleft
        z01_bd[:, 5:6] = (region_pright - region_pleft) * np.random.random([batch_size, 1]) + region_pleft
        z01_bd[:, 6:7] = (region_qright - region_qleft) * np.random.random([batch_size, 1]) + region_qleft
        z01_bd[:, 7:8] = (region_rright - region_rleft) * np.random.random([batch_size, 1]) + region_rleft

        t00_bd[:, 0:1] = (region_xright - region_xleft) * np.random.random([batch_size, 1]) + region_xleft
        t00_bd[:, 1:2] = (region_yright - region_yleft) * np.random.random([batch_size, 1]) + region_yleft
        t00_bd[:, 2:3] = (region_zright - region_zleft) * np.random.random([batch_size, 1]) + region_zleft
        t00_bd[:, 3:4] = (region_sright - region_sleft) * np.random.random([batch_size, 1]) + region_sleft
        t00_bd[:, 4:5] = region_tleft
        t00_bd[:, 5:6] = (region_pright - region_pleft) * np.random.random([batch_size, 1]) + region_pleft
        t00_bd[:, 6:7] = (region_qright - region_qleft) * np.random.random([batch_size, 1]) + region_qleft
        t00_bd[:, 7:8] = (region_rright - region_rleft) * np.random.random([batch_size, 1]) + region_rleft

        t01_bd[:, 0:1] = (region_xright - region_xleft) * np.random.random([batch_size, 1]) + region_xleft
        t01_bd[:, 1:2] = (region_yright - region_yleft) * np.random.random([batch_size, 1]) + region_yleft
        t01_bd[:, 2:3] = (region_zright - region_zleft) * np.random.random([batch_size, 1]) + region_zleft
        t01_bd[:, 3:4] = (region_sright - region_sleft) * np.random.random([batch_size, 1]) + region_sleft
        t01_bd[:, 4:5] = region_tright
        t01_bd[:, 5:6] = (region_pright - region_pleft) * np.random.random([batch_size, 1]) + region_pleft
        t01_bd[:, 6:7] = (region_qright - region_qleft) * np.random.random([batch_size, 1]) + region_qleft
        t01_bd[:, 7:8] = (region_rright - region_rleft) * np.random.random([batch_size, 1]) + region_rleft

        p00_bd[:, 0:1] = (region_xright - region_xleft) * np.random.random([batch_size, 1]) + region_xleft
        p00_bd[:, 1:2] = (region_yright - region_yleft) * np.random.random([batch_size, 1]) + region_yleft
        p00_bd[:, 2:3] = (region_zright - region_zleft) * np.random.random([batch_size, 1]) + region_zleft
        p00_bd[:, 3:4] = (region_sright - region_sleft) * np.random.random([batch_size, 1]) + region_sleft
        p00_bd[:, 4:5] = (region_tright - region_tleft) * np.random.random([batch_size, 1]) + region_tleft
        p00_bd[:, 5:6] = region_pleft
        p00_bd[:, 6:7] = (region_qright - region_qleft) * np.random.random([batch_size, 1]) + region_qleft
        p00_bd[:, 7:8] = (region_rright - region_rleft) * np.random.random([batch_size, 1]) + region_rleft

        p01_bd[:, 0:1] = (region_xright - region_xleft) * np.random.random([batch_size, 1]) + region_xleft
        p01_bd[:, 1:2] = (region_yright - region_yleft) * np.random.random([batch_size, 1]) + region_yleft
        p01_bd[:, 2:3] = (region_zright - region_zleft) * np.random.random([batch_size, 1]) + region_zleft
        p01_bd[:, 3:4] = (region_sright - region_sleft) * np.random.random([batch_size, 1]) + region_sleft
        p01_bd[:, 4:5] = (region_tright - region_tleft) * np.random.random([batch_size, 1]) + region_tleft
        p01_bd[:, 5:6] = region_pright
        p01_bd[:, 6:7] = (region_qright - region_qleft) * np.random.random([batch_size, 1]) + region_qleft
        p01_bd[:, 7:8] = (region_rright - region_rleft) * np.random.random([batch_size, 1]) + region_rleft

        q00_bd[:, 0:1] = (region_xright - region_xleft) * np.random.random([batch_size, 1]) + region_xleft
        q00_bd[:, 1:2] = (region_yright - region_yleft) * np.random.random([batch_size, 1]) + region_yleft
        q00_bd[:, 2:3] = (region_zright - region_zleft) * np.random.random([batch_size, 1]) + region_zleft
        q00_bd[:, 3:4] = (region_sright - region_sleft) * np.random.random([batch_size, 1]) + region_sleft
        q00_bd[:, 4:5] = (region_tright - region_tleft) * np.random.random([batch_size, 1]) + region_tleft
        q00_bd[:, 5:6] = (region_pright - region_pleft) * np.random.random([batch_size, 1]) + region_pleft
        q00_bd[:, 6:7] = region_qleft
        q00_bd[:, 7:8] = (region_rright - region_rleft) * np.random.random([batch_size, 1]) + region_rleft

        q01_bd[:, 0:1] = (region_xright - region_xleft) * np.random.random([batch_size, 1]) + region_xleft
        q01_bd[:, 1:2] = (region_yright - region_yleft) * np.random.random([batch_size, 1]) + region_yleft
        q01_bd[:, 2:3] = (region_zright - region_zleft) * np.random.random([batch_size, 1]) + region_zleft
        q01_bd[:, 3:4] = (region_sright - region_sleft) * np.random.random([batch_size, 1]) + region_sleft
        q01_bd[:, 4:5] = (region_tright - region_tleft) * np.random.random([batch_size, 1]) + region_tleft
        q01_bd[:, 5:6] = (region_pright - region_pleft) * np.random.random([batch_size, 1]) + region_pleft
        q01_bd[:, 6:7] = region_qright
        q01_bd[:, 7:8] = (region_rright - region_rleft) * np.random.random([batch_size, 1]) + region_rleft

        r00_bd[:, 0:1] = (region_xright - region_xleft) * np.random.random([batch_size, 1]) + region_xleft
        r00_bd[:, 1:2] = (region_yright - region_yleft) * np.random.random([batch_size, 1]) + region_yleft
        r00_bd[:, 2:3] = (region_zright - region_zleft) * np.random.random([batch_size, 1]) + region_zleft
        r00_bd[:, 3:4] = (region_sright - region_sleft) * np.random.random([batch_size, 1]) + region_sleft
        r00_bd[:, 4:5] = (region_tright - region_tleft) * np.random.random([batch_size, 1]) + region_tleft
        r00_bd[:, 5:6] = (region_pright - region_pleft) * np.random.random([batch_size, 1]) + region_pleft
        r00_bd[:, 6:7] = (region_qright - region_qleft) * np.random.random([batch_size, 1]) + region_qleft
        r00_bd[:, 7:8] = region_rleft

        r01_bd[:, 0:1] = (region_xright - region_xleft) * np.random.random([batch_size, 1]) + region_xleft
        r01_bd[:, 1:2] = (region_yright - region_yleft) * np.random.random([batch_size, 1]) + region_yleft
        r01_bd[:, 2:3] = (region_zright - region_zleft) * np.random.random([batch_size, 1]) + region_zleft
        r01_bd[:, 3:4] = (region_sright - region_sleft) * np.random.random([batch_size, 1]) + region_sleft
        r01_bd[:, 4:5] = (region_tright - region_tleft) * np.random.random([batch_size, 1]) + region_tleft
        r01_bd[:, 5:6] = (region_pright - region_pleft) * np.random.random([batch_size, 1]) + region_pleft
        r01_bd[:, 6:7] = (region_qright - region_qleft) * np.random.random([batch_size, 1]) + region_qleft
        r01_bd[:, 7:8] = region_rright

    if to_float:
        x00_bd = x00_bd.astype(np.float32)
        x01_bd = x01_bd.astype(np.float32)

        y00_bd = y00_bd.astype(np.float32)
        y01_bd = y01_bd.astype(np.float32)

        z00_bd = z00_bd.astype(np.float32)
        z01_bd = z01_bd.astype(np.float32)

        s00_bd = s00_bd.astype(np.float32)
        s01_bd = s01_bd.astype(np.float32)

        t00_bd = t00_bd.astype(np.float32)
        t01_bd = t01_bd.astype(np.float32)

        p00_bd = t00_bd.astype(np.float32)
        p01_bd = t01_bd.astype(np.float32)

        q00_bd = t00_bd.astype(np.float32)
        q01_bd = t01_bd.astype(np.float32)

        r00_bd = t00_bd.astype(np.float32)
        r01_bd = t01_bd.astype(np.float32)

    if to_torch:
        x00_bd = torch.from_numpy(x00_bd)
        x01_bd = torch.from_numpy(x01_bd)

        y00_bd = torch.from_numpy(y00_bd)
        y01_bd = torch.from_numpy(y01_bd)

        z00_bd = torch.from_numpy(z00_bd)
        z01_bd = torch.from_numpy(z01_bd)

        s00_bd = torch.from_numpy(s00_bd)
        s01_bd = torch.from_numpy(s01_bd)

        t00_bd = torch.from_numpy(t00_bd)
        t01_bd = torch.from_numpy(t01_bd)

        p00_bd = torch.from_numpy(p00_bd)
        p01_bd = torch.from_numpy(p01_bd)

        q00_bd = torch.from_numpy(q00_bd)
        q01_bd = torch.from_numpy(q01_bd)

        r00_bd = torch.from_numpy(r00_bd)
        r01_bd = torch.from_numpy(r01_bd)

        if to_cuda:
            x00_bd = x00_bd.cuda(device='cuda:' + str(gpu_no))
            x01_bd = x01_bd.cuda(device='cuda:' + str(gpu_no))

            y00_bd = y00_bd.cuda(device='cuda:' + str(gpu_no))
            y01_bd = y01_bd.cuda(device='cuda:' + str(gpu_no))

            z00_bd = z00_bd.cuda(device='cuda:' + str(gpu_no))
            z01_bd = z01_bd.cuda(device='cuda:' + str(gpu_no))

            s00_bd = s00_bd.cuda(device='cuda:' + str(gpu_no))
            s01_bd = s01_bd.cuda(device='cuda:' + str(gpu_no))

            t00_bd = t00_bd.cuda(device='cuda:' + str(gpu_no))
            t01_bd = t01_bd.cuda(device='cuda:' + str(gpu_no))

            p00_bd = p00_bd.cuda(device='cuda:' + str(gpu_no))
            p01_bd = p01_bd.cuda(device='cuda:' + str(gpu_no))

            q00_bd = q00_bd.cuda(device='cuda:' + str(gpu_no))
            q01_bd = q01_bd.cuda(device='cuda:' + str(gpu_no))

            r00_bd = r00_bd.cuda(device='cuda:' + str(gpu_no))
            r01_bd = r01_bd.cuda(device='cuda:' + str(gpu_no))

        x00_bd.requires_grad = use_grad
        x01_bd.requires_grad = use_grad

        y00_bd.requires_grad = use_grad
        y01_bd.requires_grad = use_grad

        z00_bd.requires_grad = use_grad
        z01_bd.requires_grad = use_grad

        s00_bd.requires_grad = use_grad
        s01_bd.requires_grad = use_grad

        t00_bd.requires_grad = use_grad
        t01_bd.requires_grad = use_grad

        p00_bd.requires_grad = use_grad
        p01_bd.requires_grad = use_grad

        q00_bd.requires_grad = use_grad
        q01_bd.requires_grad = use_grad

        r00_bd.requires_grad = use_grad
        r01_bd.requires_grad = use_grad

    return x00_bd, x01_bd, y00_bd, y01_bd, z00_bd, z01_bd, s00_bd, s01_bd, t00_bd, t01_bd, \
           p00_bd, p01_bd, q00_bd, q01_bd, r00_bd, r01_bd


def sampler_test():
    size = 20
    dim = 2
    left = -2
    right = 2
    x = rand_in_2D(batch_size=size, variable_dim=dim, region_left=left, region_right=right, region_bottom=left,
                   region_top=right)
    print('Endding!!!!!!')
    z = np.sin(x[:, 0]) * np.sin(x[:, 1])

    # 绘制解的3D散点图(真解和预测解)
    fig = plt.figure(figsize=(10, 10))
    ax = Axes3D(fig)
    ax.scatter(x[:, 0], x[:, 1], z, cmap='b', label='z')

    # 绘制图例
    ax.legend(loc='best')
    # 添加坐标轴(顺序是X，Y, Z)
    ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})
    ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
    ax.set_zlabel('u', fontdict={'size': 15, 'color': 'red'})
    plt.show()

    y = rand_in_2D(batch_size=size, variable_dim=dim,region_left=left, region_right=right, region_bottom=left,
                   region_top=right)
    print('Endding!!!!!!')


def sample_test2():
    size = 2000
    dim = 2
    left = -2
    right = 2
    x = rand_in_2D(batch_size=size, variable_dim=dim, region_left=left, region_right=right, region_bottom=left,
                   region_top=right)
    print('Endding!!!!!!')
    z = np.sin(x[:, 0]) * np.sin(x[:, 1])

    # plt.figure()
    # plt.plot(x[:, 0], x[:, 1], 'b*')
    # plt.xlabel('x', fontsize=14)
    # plt.ylabel('y', fontsize=14)
    # plt.legend(fontsize=18)
    # plt.title('point2lhs')
    # plt.show()

    y = rand_it(batch_size=size, variable_dim=dim, region_a=left, region_b=right)
    print('Endding!!!!!!')
    plt.figure()
    plt.plot(x[:, 0], x[:, 1], 'b*', label='lhs')
    plt.plot(y[:, 0], y[:, 1], 'r.', label='rand')
    plt.xlabel('x', fontsize=14)
    plt.ylabel('y', fontsize=14)
    plt.legend(fontsize=18, loc='lower left')
    plt.show()


def sample_test_1D_sobol():
    size = 200
    dim = 1
    left = -2
    right = 2
    xy = rand_it_Sobol(batch_size=size, variable_dim=2, region_a=left, region_b=right)

    xy2 = rand_it_Sobol(batch_size=size, variable_dim=2, region_a=left, region_b=right)

    xy3 = rand_it_Sobol(batch_size=size, variable_dim=2, region_a=left, region_b=right)

    xy4 = rand_it_Sobol(batch_size=size, variable_dim=2, region_a=left, region_b=right)

    xy5 = rand_it_Sobol(batch_size=size, variable_dim=2, region_a=left, region_b=right)
    print('Endding!!!!!!')

    plt.figure()

    plt.plot(xy[:, 0], xy[:, 1], 'r.', label='lhs2D')
    plt.plot(xy2[:, 0], xy2[:, 1], 'm.', label='lhs2D')
    plt.plot(xy3[:, 0], xy3[:, 1], 'b*', label='lhs1D')
    plt.plot(xy4[:, 0], xy4[:, 1], 'g*', label='lhs1D')

    plt.plot(xy5[:, 0], xy4[:, 1], 'm*', label='lhs1D')
    plt.xlabel('x', fontsize=14)
    plt.ylabel('y', fontsize=14)
    plt.legend(fontsize=18, loc='lower left')
    plt.show()


def lhs_sample_bd_test_2D():
    size = 2000
    dim = 2
    left = -2
    right = 2
    xy_left, xy_right, xy_bottom, xy_top = rand_bd_2D(batch_size=size, variable_dim=dim, region_left=left,
                                                      region_right=right, region_bottom=left, region_top=right)
    print('Endding!!!!!!')

    plt.figure()
    plt.plot(xy_left[:, 0], xy_left[:, 1], 'b*', label='lhs')
    plt.plot(xy_right[:, 0], xy_right[:, 1], 'r.', label='rand')
    plt.xlabel('x', fontsize=14)
    plt.ylabel('y', fontsize=14)
    plt.legend(fontsize=18, loc='lower left')
    plt.show()


def lhs_sample_bd_test_3D():
    size = 2000
    dim = 3
    left = -2
    right = 2
    xyz_left_batch, xyz_right_batch, xyz_front_batch, xyz_behind_batch, xyz_bottom_batch, xyz_top_batch = rand_bd_3D(
        batch_size=size, variable_dim=dim, region_left=left, region_right=right,
            region_behind=left, region_front=right, region_bottom=left, region_top=right, to_torch=False)
    print('Endding!!!!!!')


def lhs_sample_test_3D():
    size = 2000
    dim = 3
    left = -2
    right = 2
    xyz = rand_in_3D(
        batch_size=size, variable_dim=dim, region_left=left, region_right=right,
            region_behind=left, region_front=right, region_bottom=left, region_top=right, to_torch=False)
    print('Endding!!!!!!')


def sobol_sample_bd_test_2D():
    size = 2000
    dim = 2
    left = -2
    right = 2
    xy_left, xy_right, xy_bottom, xy_top = rand_bd_2D_sobol(batch_size=size, variable_dim=dim,
                                                            region_a=left, region_b=right)
    print('Endding!!!!!!')

    plt.figure()
    plt.plot(xy_left[:, 0], xy_left[:, 1], 'b*', label='lhs')
    plt.plot(xy_right[:, 0], xy_right[:, 1], 'r.', label='rand')
    plt.xlabel('x', fontsize=14)
    plt.ylabel('y', fontsize=14)
    plt.legend(fontsize=18, loc='lower left')
    plt.show()


if __name__ == "__main__":
    # sample_test2()
    # sample_test_1D()
    # sample_test_1D_sobol()
    # sample_test_2D()

    # lhs_sample_bd_test_2D()

    lhs_sample_bd_test_3D()

    # lhs_sample_test_3D()

    # sobol_sample_bd_test_2D()

    # aa = np.random.randn(2, 4)
    # print(aa)