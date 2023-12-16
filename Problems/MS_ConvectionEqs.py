import numpy as np


def get_infos2Convection_1D(in_dim=1, out_dim=1, region_a=0.0, region_b=1.0, index2p=2, eps=0.01, eqs_name=None):
    if eqs_name == 'Boltzmann1':
        llam = 20
        mu = 50
        f = lambda x: (llam*llam+mu*mu)*np.sin(x)
        Aeps = lambda x: 1.0*np.ones_like(x)
        kappa = lambda x: llam*llam*np.ones_like(x)
        utrue = lambda x: -1.0*(np.sin(mu)/np.sinh(llam))*np.sinh(llam*x) + np.sin(mu*x)
        ul = lambda x: np.zeros_like(x)
        ur = lambda x: np.zeros_like(x)
        return Aeps, kappa, utrue, ul, ur, f
    elif eqs_name == 'Boltzmann2':
        kappa = lambda x: np.ones_like(x)
        Aeps = lambda x: 1.0 / (2 + np.cos(2 * np.pi * x / eps))

        utrue = lambda x: x - np.square(x) + (eps / (4*np.pi)) * np.sin(np.pi * 2 * x / eps)

        ul = lambda x: np.zeros_like(x)

        ur = lambda x: np.zeros_like(x)

        if index2p == 2:
            f = lambda x: 2.0/(2 + np.cos(2 * np.pi * x / eps)) + (4*np.pi*x/eps)*np.sin(np.pi * 2 * x / eps)/\
                          ((2 + np.cos(2 * np.pi * x / eps))*(2 + np.cos(2 * np.pi * x / eps))) + x - np.square(x) \
                          + (eps / (4*np.pi)) * np.sin(np.pi * 2 * x / eps)

        return Aeps, kappa, utrue, ul, ur, f


def get_infos2Convection_2D(equa_name=None, eps=0.1, region_lb=0.1, region_rt=1.0):
    if equa_name == 'Convection2':
        f = lambda x, y: eps * ((np.pi) ** 2) * (
                np.sin(np.pi * x) * np.sin(np.pi * y) + 5 * np.sin(10 * np.pi * x) * np.sin(10 * np.pi * y)) + \
                         np.cos(18 * np.pi * y) * np.sin(18 * np.pi * x) * \
                         (0.5 * np.pi * np.cos(np.pi * x) * np.sin(np.pi * y) + 0.25 * np.pi * np.cos(
                             10 * np.pi * x) * np.sin(10 * np.pi * y)) - \
                         np.cos(18 * np.pi * x) * np.sin(18 * np.pi * y) * \
                         (0.5 * np.pi * np.sin(np.pi * x) * np.cos(np.pi * y) + 0.25 * np.pi * np.sin(
                             10 * np.pi * x) * np.cos(10 * np.pi * y))
        A_eps = lambda x, y: eps * np.ones_like(x)
        u = lambda x, y: 0.5 * np.sin(np.pi * x) * np.sin(np.pi * y) + 0.025 * np.sin(10 * np.pi * x) * np.sin(
            10 * np.pi * y)
        bx = lambda x, y: np.cos(18 * np.pi * y) * np.sin(18 * np.pi * x)
        by = lambda x, y: -np.cos(18 * np.pi * x) * np.sin(18 * np.pi * y)
        ux_left = lambda x, y: 0.5 * np.sin(np.pi * region_lb) * np.sin(np.pi * y) + 0.025 * np.sin(
            10 * np.pi * region_lb) * np.sin(10 * np.pi * y)
        ux_right = lambda x, y: 0.5 * np.sin(np.pi * region_rt) * np.sin(np.pi * y) + 0.025 * np.sin(
            10 * np.pi * region_rt) * np.sin(10 * np.pi * y)
        uy_bottom = lambda x, y: 0.5 * np.sin(np.pi * x) * np.sin(np.pi * region_lb) + 0.025 * np.sin(
            10 * np.pi * x) * np.sin(10 * np.pi * region_lb)
        uy_top = lambda x, y: 0.5 * np.sin(np.pi * x) * np.sin(np.pi * region_rt) + 0.025 * np.sin(
            10 * np.pi * x) * np.sin(10 * np.pi * region_rt)

        return A_eps, bx, by, u, ux_left, ux_right, uy_top, uy_bottom, f


def get_infos2Convection_3D(input_dim=1, out_dim=1, mesh_number=2, intervalL=0.0, intervalR=1.0, equa_name=None):
    if equa_name == 'Boltzmann1':
        # mu1= 2*np.pi
        # mu2 = 4*np.pi
        # mu3 = 8*np.pi
        # mu1 = np.pi
        # mu2 = 5 * np.pi
        # mu3 = 10 * np.pi
        mu1 = np.pi
        mu2 = 10 * np.pi
        mu3 = 20 * np.pi
        f = lambda x, y, z: (mu1*mu1+mu2*mu2+mu3*mu3+x*x+2*y*y+3*z*z)*np.sin(mu1*x)*np.sin(mu2*y)*np.sin(mu3*z)
        A_eps = lambda x, y, z: 1.0*np.ones_like(x)
        kappa = lambda x, y, z: x*x+2*y*y+3*z*z
        u = lambda x, y, z: np.sin(mu1*x)*np.sin(mu2*y)*np.sin(mu3*z)
        u_00 = lambda x, y, z: np.sin(mu1*intervalL)*np.sin(mu2*y)*np.sin(mu3*z)
        u_01 = lambda x, y, z: np.sin(mu1*intervalR)*np.sin(mu2*y)*np.sin(mu3*z)
        u_10 = lambda x, y, z: np.sin(mu1*x)*np.sin(mu2*intervalL)*np.sin(mu3*z)
        u_11 = lambda x, y, z: np.sin(mu1*x)*np.sin(mu2*intervalR)*np.sin(mu3*z)
        u_20 = lambda x, y, z: np.sin(mu1*x)*np.sin(mu2*y)*np.sin(mu3*intervalL)
        u_21 = lambda x, y, z: np.sin(mu1*x)*np.sin(mu2*y)*np.sin(mu3*intervalR)

    return A_eps, kappa, f, u, u_00, u_01, u_10, u_11, u_20, u_21


def get_infos2Convection_5D(input_dim=1, out_dim=1, mesh_number=2, intervalL=0.0, intervalR=1.0, equa_name=None):
    if equa_name == 'Boltzmann1':
        lam = 2
        mu = 30
        f = lambda x, y: (lam*lam+mu*mu)*(np.sin(mu*x) + np.sin(mu*y))
        A_eps = lambda x, y: 1.0*np.ones_like(x)
        kappa = lambda x, y: lam*lam*np.ones_like(x)
        u = lambda x, y: -1.0*(np.sin(mu)/np.sinh(lam))*np.sinh(lam*x) + np.sin(mu*x) -1.0*(np.sin(mu)/np.sinh(lam))*np.sinh(lam*y) + np.sin(mu*y)
        u_00 = lambda x, y, z, s, t: np.zeros_like(x)
        u_01 = lambda x, y, z, s, t: np.zeros_like(x)
        u_10 = lambda x, y, z, s, t: np.zeros_like(x)
        u_11 = lambda x, y, z, s, t: np.zeros_like(x)
        u_20 = lambda x, y, z, s, t: np.zeros_like(x)
        u_21 = lambda x, y, z, s, t: np.zeros_like(x)
        u_30 = lambda x, y, z, s, t: np.zeros_like(x)
        u_31 = lambda x, y, z, s, t: np.zeros_like(x)
        u_40 = lambda x, y, z, s, t: np.zeros_like(x)
        u_41 = lambda x, y, z, s, t: np.zeros_like(x)

    return A_eps, kappa, u, f, u_00, u_01, u_10, u_11, u_20, u_21, u_30, u_31, u_40, u_41