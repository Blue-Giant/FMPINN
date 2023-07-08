import numpy as torch
import torch


def get_infos2Laplace_1D(input_dim=1, out_dim=1, intervalL=0.0, intervalR=1.0, equa_name=None):
    # -uxx = f
    if equa_name == 'PDE1':
        # u=sin(pi*x), f=-pi*pi*sin(pi*x)
        fside = lambda x: -(torch.pi)*(torch.pi)*torch.sin(torch.pi*x)
        utrue = lambda x: torch.sin(torch.pi*x)
        uleft = lambda x: torch.sin(torch.pi*x)
        uright = lambda x: torch.sin(torch.pi*x)
    return fside, utrue, uleft, uright


# 偏微分方程的一些信息:边界条件，初始条件，真解，右端项函数
def get_infos2Laplace_2D(input_dim=1, out_dim=1, left_bottom=0.0, right_top=1.0, equa_name=None):
    if equa_name == 'PDE1':
        # u=exp(-x)(x_y^3), f = -exp(-x)(x-2+y^3+6y)
        f_side = lambda x, y: -(torch.exp(-1.0*x)) * (x - 2 + torch.pow(y, 3) + 6 * y)

        u_true = lambda x, y: (torch.exp(-1.0*x))*(x + torch.pow(y, 3))

        ux_left = lambda x, y: (torch.exp(-1.0*x))*(x + torch.pow(y, 3))
        ux_right = lambda x, y: (torch.exp(-1.0*x))*(x + torch.pow(y, 3))
        uy_bottom = lambda x, y: (torch.exp(-1.0*x))*(x + torch.pow(y, 3))
        uy_top = lambda x, y: (torch.exp(-1.0*x))*(x + torch.pow(y, 3))
        return f_side, u_true, ux_left, ux_right, uy_bottom, uy_top
    elif equa_name == 'PDE2':
        f_side = lambda x, y: (-1.0)*torch.sin(torch.pi*x) * (2 - torch.square(torch.pi)*torch.square(y))

        u_true = lambda x, y: torch.square(y)*torch.sin(torch.pi*x)

        ux_left = lambda x, y: torch.square(y) * torch.sin(torch.pi * left_bottom)
        ux_right = lambda x, y: torch.square(y) * torch.sin(torch.pi * right_top)
        uy_bottom = lambda x, y: torch.square(left_bottom) * torch.sin(torch.pi * x)
        uy_top = lambda x, y: torch.square(right_top) * torch.sin(torch.pi * x)
        return f_side, u_true, ux_left, ux_right, uy_bottom, uy_top
    elif equa_name == 'PDE3':
        # u=exp(x+y), f = -2*exp(x+y)
        f_side = lambda x, y: -2.0*(torch.exp(x)*torch.exp(y))
        u_true = lambda x, y: torch.exp(x)*torch.exp(y)
        ux_left = lambda x, y: torch.multiply(torch.exp(y), torch.exp(left_bottom))
        ux_right = lambda x, y: torch.multiply(torch.exp(y), torch.exp(right_top))
        uy_bottom = lambda x, y: torch.multiply(torch.exp(x), torch.exp(left_bottom))
        uy_top = lambda x, y: torch.multiply(torch.exp(x), torch.exp(right_top))
        return f_side, u_true, ux_left, ux_right, uy_bottom, uy_top
    elif equa_name == 'PDE4':
        # u=(1/4)*(x^2+y^2), f = -1
        f_side = lambda x, y: -1.0*torch.ones_like(x)
        u_true = lambda x, y: 0.25*(torch.pow(x, 2)+torch.pow(y, 2))
        ux_left = lambda x, y: 0.25 * torch.pow(y, 2) + 0.25 * torch.pow(left_bottom, 2)
        ux_right = lambda x, y: 0.25 * torch.pow(y, 2) + 0.25 * torch.pow(right_top, 2)
        uy_bottom = lambda x, y: 0.25 * torch.pow(x, 2) + 0.25 * torch.pow(left_bottom, 2)
        uy_top = lambda x, y: 0.25 * torch.pow(x, 2) + 0.25 * torch.pow(right_top, 2)
        return f_side, u_true, ux_left, ux_right, uy_bottom, uy_top
    elif equa_name == 'PDE5':
        # u=(1/4)*(x^2+y^2)+x+y, f = -1
        f_side = lambda x, y: -1.0*torch.ones_like(x)

        u_true = lambda x, y: 0.25*(torch.pow(x, 2)+torch.pow(y, 2)) + x + y

        ux_left = lambda x, y: 0.25 * torch.pow(y, 2) + 0.25 * torch.pow(left_bottom, 2) + left_bottom + y
        ux_right = lambda x, y: 0.25 * torch.pow(y, 2) + 0.25 * torch.pow(right_top, 2) + right_top + y
        uy_bottom = lambda x, y: 0.25 * torch.pow(x, 2) + torch.pow(left_bottom, 2) + left_bottom + x
        uy_top = lambda x, y: 0.25 * torch.pow(x, 2) + 0.25 * torch.pow(right_top, 2) + right_top + x
        return f_side, u_true, ux_left, ux_right, uy_bottom, uy_top
    elif equa_name == 'PDE6':
        # u=(1/2)*(x^2)*(y^2), f = -(x^2+y^2)
        f_side = lambda x, y: -1.0*(torch.pow(x, 2)+torch.pow(y, 2))

        u_true = lambda x, y: 0.5 * (torch.pow(x, 2) * torch.pow(y, 2))

        ux_left = lambda x, y: 0.5 * (torch.pow(left_bottom, 2) * torch.pow(y, 2))
        ux_right = lambda x, y: 0.5 * (torch.pow(right_top, 2) * torch.pow(y, 2))
        uy_bottom = lambda x, y: 0.5 * (torch.pow(x, 2) * torch.pow(left_bottom, 2))
        uy_top = lambda x, y: 0.5 * (torch.pow(x, 2) * torch.pow(right_top, 2))
        return f_side, u_true, ux_left, ux_right, uy_bottom, uy_top
    elif equa_name == 'PDE7':
        # u=(1/2)*(x^2)*(y^2)+x+y, f = -(x^2+y^2)
        f_side = lambda x, y: -1.0*(torch.pow(x, 2)+torch.pow(y, 2))

        u_true = lambda x, y: 0.5*(torch.pow(x, 2)*torch.pow(y, 2)) + x*torch.ones_like(x) + y*torch.ones_like(y)

        ux_left = lambda x, y: 0.5 * torch.multiply(torch.pow(left_bottom, 2), torch.pow(y, 2)) + left_bottom + y
        ux_right = lambda x, y: 0.5 * torch.multiply(torch.pow(right_top, 2), torch.pow(y, 2)) + right_top + y
        uy_bottom = lambda x, y: 0.5 * torch.multiply(torch.pow(x, 2), torch.pow(left_bottom, 2)) + x + left_bottom
        uy_top = lambda x, y: 0.5 * torch.multiply(torch.pow(x, 2), torch.pow(right_top, 2)) + x + right_top
        return f_side, u_true, ux_left, ux_right, uy_bottom, uy_top


# 偏微分方程的一些信息:边界条件，初始条件，真解，右端项函数
def get_infos2Laplace_3D(input_dim=1, out_dim=1, intervalL=0.0, intervalR=1.0, equa_name=None):
    if equa_name == 'PDE1':
        # -Laplace U = f
        # u=sin(pi*x)*sin(pi*y)*sin(pi*z), f=-pi*pi*sin(pi*x)*sin(pi*y)*sin(pi*z)
        fside = lambda x, y, z: -(torch.pi)*(torch.pi)*torch.sin(torch.pi*x)
        utrue = lambda x, y, z: torch.sin(torch.pi*x)*torch.sin(torch.pi*y)*torch.sin(torch.pi*z)
        u_00 = lambda x, y, z: torch.sin(torch.pi*x)*torch.sin(torch.pi*y)*torch.sin(torch.pi*z)
        u_01 = lambda x, y, z: torch.sin(torch.pi*x)*torch.sin(torch.pi*y)*torch.sin(torch.pi*z)
        u_10 = lambda x, y, z: torch.sin(torch.pi*x)*torch.sin(torch.pi*y)*torch.sin(torch.pi*z)
        u_11 = lambda x, y, z: torch.sin(torch.pi*x)*torch.sin(torch.pi*y)*torch.sin(torch.pi*z)
        u_20 = lambda x, y, z: torch.sin(torch.pi*x)*torch.sin(torch.pi*y)*torch.sin(torch.pi*z)
        u_21 = lambda x, y, z: torch.sin(torch.pi*x)*torch.sin(torch.pi*y)*torch.sin(torch.pi*z)
    return fside, utrue, u_00, u_01, u_10, u_11, u_20, u_21


# 偏微分方程的一些信息:边界条件，初始条件，真解，右端项函数
def get_infos2Laplace_5D(input_dim=1, out_dim=1, intervalL=0.0, intervalR=1.0, equa_name=None):
    if equa_name == 'PDE1':
        # u=sin(pi*x), f=-pi*pi*sin(pi*x)
        fside = lambda x, y, z, s, t: -(torch.pi)*(torch.pi)*torch.sin(torch.pi*x)
        utrue = lambda x, y, z, s, t: torch.sin(torch.pi*x)*torch.sin(torch.pi*y)*torch.sin(torch.pi*z)*torch.sin(torch.pi*s)*torch.sin(torch.pi*t)
        u_00 = lambda x, y, z, s, t: torch.sin(torch.pi*intervalL)*torch.sin(torch.pi*y)*torch.sin(torch.pi*z)*torch.sin(torch.pi*s)*torch.sin(torch.pi*t)
        u_01 = lambda x, y, z, s, t: torch.sin(torch.pi*intervalR)*torch.sin(torch.pi*y)*torch.sin(torch.pi*z)*torch.sin(torch.pi*s)*torch.sin(torch.pi*t)
        u_10 = lambda x, y, z, s, t: torch.sin(torch.pi * x) * torch.sin(torch.pi * intervalL) * torch.sin(torch.pi * z) * torch.sin(torch.pi * s) * torch.sin(torch.pi * t)
        u_11 = lambda x, y, z, s, t: torch.sin(torch.pi * x) * torch.sin(torch.pi * intervalR) * torch.sin(torch.pi * z) * torch.sin(torch.pi * s) * torch.sin(torch.pi * t)
        u_20 = lambda x, y, z, s, t: torch.sin(torch.pi * x) * torch.sin(torch.pi * y) * torch.sin(torch.pi * intervalL) * torch.sin(torch.pi * s) * torch.sin(torch.pi * t)
        u_21 = lambda x, y, z, s, t: torch.sin(torch.pi * x) * torch.sin(torch.pi * y) * torch.sin(torch.pi * intervalR) * torch.sin(torch.pi * s) * torch.sin(torch.pi * t)
        u_30 = lambda x, y, z, s, t: torch.sin(torch.pi * x) * torch.sin(torch.pi * y) * torch.sin(torch.pi * z) * torch.sin(torch.pi * intervalL) * torch.sin(torch.pi * t)
        u_31 = lambda x, y, z, s, t: torch.sin(torch.pi * x) * torch.sin(torch.pi * y) * torch.sin(torch.pi * z) * torch.sin(torch.pi * intervalR) * torch.sin(torch.pi * t)
        u_40 = lambda x, y, z, s, t: torch.sin(torch.pi * x) * torch.sin(torch.pi * y) * torch.sin(torch.pi * z) * torch.sin(torch.pi * s) * torch.sin(torch.pi * intervalL)
        u_41 = lambda x, y, z, s, t: torch.sin(torch.pi * x) * torch.sin(torch.pi * y) * torch.sin(torch.pi * z) * torch.sin(torch.pi * s) * torch.sin(torch.pi * intervalR)
    return fside, utrue, u_00, u_01, u_10, u_11, u_20, u_21, u_30, u_31, u_40, u_41