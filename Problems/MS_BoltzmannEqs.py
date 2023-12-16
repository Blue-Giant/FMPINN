import numpy as np


def get_infos2Boltzmann_1D(in_dim=1, out_dim=1, region_a=0.0, region_b=1.0, index2p=2, eps=0.01, eqs_name=None):
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

        utrue = lambda x: x - np.mul(x, x) + (eps / (4*np.pi)) * np.sin(np.pi * 2 * x / eps)

        ul = lambda x: np.zeros_like(x)

        ur = lambda x: np.zeros_like(x)

        if index2p == 2:
            f = lambda x: 2.0/(2 + np.cos(2 * np.pi * x / eps)) + (4*np.pi*x/eps)*np.sin(np.pi * 2 * x / eps)/\
                          ((2 + np.cos(2 * np.pi * x / eps))*(2 + np.cos(2 * np.pi * x / eps))) + x - np.square(x) \
                          + (eps / (4*np.pi)) * np.sin(np.pi * 2 * x / eps)

        return Aeps, kappa, utrue, ul, ur, f


def get_infos2Boltzmann_2D(equa_name=None, intervalL=0.1, intervalR=1.0):
    if equa_name == 'Boltzmann1':
        lam = 2
        mu = 30
        f = lambda x, y: (lam*lam+mu*mu)*(np.sin(mu*x) + np.sin(mu*y))
        A_eps = lambda x, y: 1.0*np.ones_like(x)
        kappa = lambda x, y: lam*lam*np.ones_like(x)
        u = lambda x, y: -1.0*(np.sin(mu)/np.sinh(lam))*np.sinh(lam*x) + np.sin(mu*x) -1.0*(np.sin(mu)/np.sinh(lam))*np.sinh(lam*y) + np.sin(mu*y)
        ux_left = lambda x, y: np.zeros_like(x)
        ux_right = lambda x, y: np.zeros_like(x)
        uy_bottom = lambda x, y: np.zeros_like(x)
        uy_top = lambda x, y: np.zeros_like(x)
    elif equa_name == 'Boltzmann2':
        # lam = 20
        # mu = 50
        f = lambda x, y: 5 * ((np.pi) ** 2) * (0.5 * np.sin(np.pi * x) * np.cos(np.pi * y) + 0.25 * np.sin(10 * np.pi * x) * np.cos(10 * np.pi * y)) * \
                        (0.25 * np.cos(5 * np.pi * x) * np.sin(10 * np.pi * y) + 0.5 * np.cos(15 * np.pi * x) * np.sin(20 * np.pi * y)) + \
                        5 * ((np.pi) ** 2) * (0.5 * np.cos(np.pi * x) * np.sin(np.pi * y) + 0.25 * np.cos(10 * np.pi * x) * np.sin(10 * np.pi * y)) * \
                        (0.125 * np.sin(5 * np.pi * x) * np.cos(10 * np.pi * y) + 0.125 * 3 * np.sin(15 * np.pi * x) * np.cos(20 * np.pi * y)) + \
                        ((np.pi) ** 2) * (np.sin(np.pi * x) * np.sin(np.pi * y) + 5 * np.sin(10 * np.pi * x) * np.sin(10 * np.pi * y)) * \
                        (0.125 * np.cos(5 * np.pi * x) * np.cos(10 * np.pi * y) + 0.125 * np.cos(15 * np.pi * x) * np.cos(20 * np.pi * y) + 0.5) + \
                         0.5 *np.pi*np.pi* np.sin(np.pi * x) * np.sin(np.pi * y) + 0.025 *np.pi*np.pi* np.sin(10 * np.pi * x) * np.sin(10 * np.pi * y)

        A_eps = lambda x, y: 0.5 + 0.125*np.cos(5*np.pi*x)*np.cos(10*np.pi*y) + 0.125*np.cos(15*np.pi*x)*np.cos(20*np.pi*y)
        kappa = lambda x, y: np.pi*np.pi*np.ones_like(x)
        u = lambda x, y: 0.5*np.sin(np.pi*x)*np.sin(np.pi*y)+0.025*np.sin(10*np.pi*x)*np.sin(10*np.pi*y)
        ux_left = lambda x, y: np.zeros_like(x)
        ux_right = lambda x, y: np.zeros_like(x)
        uy_bottom = lambda x, y: np.zeros_like(x)
        uy_top = lambda x, y: np.zeros_like(x)
    elif equa_name == 'Boltzmann3':
        f = lambda x, y: np.ones_like(x)
        A_eps = lambda x, y: 0.5 + 0.25*np.cos(np.pi*x)*np.cos(np.pi*y) + 0.25*np.cos(20*np.pi*x)*np.cos(20*np.pi*y)
        kappa = lambda x, y: np.pi*np.pi*np.ones_like(x)
        u = lambda x, y: np.sin(np.pi*x)*np.sin(np.pi*y)+0.05*np.sin(20*np.pi*x)*np.sin(20*np.pi*y)
        ux_left = lambda x, y: np.zeros_like(x)
        ux_right = lambda x, y: np.zeros_like(x)
        uy_bottom = lambda x, y: np.zeros_like(x)
        uy_top = lambda x, y: np.zeros_like(x)
    elif equa_name == 'Boltzmann4':
        f = lambda x, y: np.ones_like(x)
        A_eps = lambda x, y: 0.5 + 0.25*np.cos(np.pi*x)*np.cos(np.pi*y) + 0.25*np.cos(10*np.pi*x)*np.cos(10*np.pi*y) + 0.25*np.cos(20*np.pi*x)*np.cos(20*np.pi*y)
        kappa = lambda x, y: np.pi*np.pi*np.ones_like(x)
        u = lambda x, y: np.sin(np.pi*x)*np.sin(np.pi*y)+0.05*np.sin(10*np.pi*x)*np.sin(10*np.pi*y)+0.01*np.sin(20*np.pi*x)*np.sin(20*np.pi*y)
        ux_left = lambda x, y: np.zeros_like(x)
        ux_right = lambda x, y: np.zeros_like(x)
        uy_bottom = lambda x, y: np.zeros_like(x)
        uy_top = lambda x, y: np.zeros_like(x)
    elif equa_name == 'Boltzmann5':
        f = lambda x, y: np.ones_like(x)
        A_eps = lambda x, y: 0.5 + 0.25*np.cos(np.pi*x)*np.cos(np.pi*y) + 0.25*np.cos(10*np.pi*x)*np.cos(10*np.pi*y) + 0.25*np.cos(20*np.pi*x)*np.cos(20*np.pi*y)
        kappa = lambda x, y: np.pi*np.pi*np.ones_like(x)
        u = lambda x, y: np.sin(np.pi*x)*np.sin(np.pi*y)+0.05*np.sin(20*np.pi*x)*np.sin(20*np.pi*y)
        ux_left = lambda x, y: np.zeros_like(x)
        ux_right = lambda x, y: np.zeros_like(x)
        uy_bottom = lambda x, y: np.zeros_like(x)
        uy_top = lambda x, y: np.zeros_like(x)
    elif equa_name == 'Boltzmann6':
        f = lambda x, y: np.ones_like(x)
        A_eps = lambda x, y: 0.5 + 0.25*np.cos(np.pi*x)*np.cos(np.pi*y) + 0.25*np.cos(20*np.pi*x)*np.cos(20*np.pi*y)
        kappa = lambda x, y: np.pi*np.pi*np.ones_like(x)
        u = lambda x, y: np.sin(np.pi*x)*np.sin(np.pi*y)+0.05*np.sin(10*np.pi*x)*np.sin(10*np.pi*y)+0.01*np.sin(20*np.pi*x)*np.sin(20*np.pi*y)
        ux_left = lambda x, y: np.zeros_like(x)
        ux_right = lambda x, y: np.zeros_like(x)
        uy_bottom = lambda x, y: np.zeros_like(x)
        uy_top = lambda x, y: np.zeros_like(x)

    return A_eps, kappa, u, ux_left, ux_right, uy_top, uy_bottom, f


def get_foreside2Boltzmann2D(x=None, y=None, equa_name='Boltzmann3'):
    if equa_name == 'Boltzmann3':
        u = np.sin(np.pi*x)*np.sin(np.pi*y)+0.05*np.sin(20*np.pi*x)*np.sin(20*np.pi*y)
        Aeps = 0.5+0.25*np.cos(np.pi*x)*np.cos(np.pi*y)+0.25*np.cos(20*np.pi*x)*np.cos(20*np.pi*y)

        ux = 1.0*np.pi*np.cos(np.pi*x)*np.sin(np.pi*y)+1.0*np.pi*np.cos(20*np.pi*x)*np.sin(20*np.pi*y)
        uy = 1.0*np.pi*np.sin(np.pi*x)*np.cos(np.pi*y)+1.0*np.pi*np.sin(20*np.pi*x)*np.cos(20*np.pi*y)
        uxx = -1.0*np.pi*np.pi*np.sin(np.pi*x)*np.sin(np.pi*y)-20*np.pi*np.pi*np.sin(20*np.pi*x)*np.sin(20*np.pi*y)
        uyy = -1.0*np.pi*np.pi*np.sin(np.pi*x)*np.sin(np.pi*y)-20*np.pi*np.pi*np.sin(20*np.pi*x)*np.sin(20*np.pi*y)

        Aepsx = -0.25*np.pi*np.sin(np.pi*x)*np.cos(np.pi*y)-0.25*20*np.pi*np.sin(20*np.pi*x)*np.cos(20*np.pi*y)
        Aepsy = -0.25*np.pi*np.cos(np.pi*x)*np.sin(np.pi*y)-0.25*20*np.pi*np.cos(20*np.pi*x)*np.sin(20*np.pi*y)

        fside = -1.0*(Aepsx*ux + Aepsy*uy + Aeps*(uxx+uyy)) + 1.0*np.pi*np.pi*u
    elif equa_name == 'Boltzmann4':
        u = np.sin(np.pi*x)*np.sin(np.pi*y)+0.05*np.sin(10*np.pi*x)*np.sin(10*np.pi*y)+0.01*np.sin(20*np.pi*x)*np.sin(20*np.pi*y)
        Aeps = 0.5+0.25*np.cos(np.pi*x)*np.cos(np.pi*y)+0.25*np.cos(10*np.pi*x)*np.cos(10*np.pi*y)+0.25*np.cos(20*np.pi*x)*np.cos(20*np.pi*y)

        ux = 1.0*np.pi*np.cos(np.pi*x)*np.sin(np.pi*y)+0.5*np.pi*np.cos(10*np.pi*x)*np.sin(10*np.pi*y)+0.2*np.pi*np.cos(20*np.pi*x)*np.sin(20*np.pi*y)
        uy = 1.0*np.pi*np.sin(np.pi*x)*np.cos(np.pi*y)+0.5*np.pi*np.sin(10*np.pi*x)*np.cos(10*np.pi*y)+0.2*np.pi*np.sin(20*np.pi*x)*np.cos(20*np.pi*y)
        uxx = -1.0*np.pi*np.pi*np.sin(np.pi*x)*np.sin(np.pi*y)-5.0*np.pi*np.pi*np.sin(10*np.pi*x)*np.sin(10*np.pi*y)-4.0*np.pi*np.pi*np.sin(20*np.pi*x)*np.sin(20*np.pi*y)
        uyy = -1.0*np.pi*np.pi*np.sin(np.pi*x)*np.sin(np.pi*y)-5.0*np.pi*np.pi*np.sin(10*np.pi*x)*np.sin(10*np.pi*y)-4.0*np.pi*np.pi*np.sin(20*np.pi*x)*np.sin(20*np.pi*y)

        Aepsx = -0.25*np.pi*np.sin(np.pi*x)*np.cos(np.pi*y)-0.25*10*np.pi*np.sin(10*np.pi*x)*np.cos(10*np.pi*y)-0.25*20*np.pi*np.sin(20*np.pi*x)*np.cos(20*np.pi*y)
        Aepsy = -0.25*np.pi*np.cos(np.pi*x)*np.sin(np.pi*y)-0.25*10*np.pi*np.cos(10*np.pi*x)*np.sin(10*np.pi*y)-0.25*20*np.pi*np.cos(20*np.pi*x)*np.sin(20*np.pi*y)

        fside = -1.0*(Aepsx*ux + Aepsy*uy + Aeps*(uxx+uyy)) + 1.0*np.pi*np.pi*u
    elif equa_name == 'Boltzmann5':
        u = np.sin(np.pi*x)*np.sin(np.pi*y)+0.05*np.sin(20*np.pi*x)*np.sin(20*np.pi*y)
        Aeps = 0.5+0.25*np.cos(np.pi*x)*np.cos(np.pi*y)+0.25*np.cos(10*np.pi*x)*np.cos(10*np.pi*y)+0.25*np.cos(20*np.pi*x)*np.cos(20*np.pi*y)

        ux = 1.0*np.pi*np.cos(np.pi*x)*np.sin(np.pi*y)+1.0*np.pi*np.cos(20*np.pi*x)*np.sin(20*np.pi*y)
        uy = 1.0*np.pi*np.sin(np.pi*x)*np.cos(np.pi*y)+1.0*np.pi*np.sin(20*np.pi*x)*np.cos(20*np.pi*y)
        uxx = -1.0*np.pi*np.pi*np.sin(np.pi*x)*np.sin(np.pi*y)-20*np.pi*np.pi*np.sin(20*np.pi*x)*np.sin(20*np.pi*y)
        uyy = -1.0*np.pi*np.pi*np.sin(np.pi*x)*np.sin(np.pi*y)-20*np.pi*np.pi*np.sin(20*np.pi*x)*np.sin(20*np.pi*y)

        Aepsx = -0.25*np.pi*np.sin(np.pi*x)*np.cos(np.pi*y)-0.25*10*np.pi*np.sin(10*np.pi*x)*np.cos(10*np.pi*y)-0.25*20*np.pi*np.sin(20*np.pi*x)*np.cos(20*np.pi*y)
        Aepsy = -0.25*np.pi*np.cos(np.pi*x)*np.sin(np.pi*y)-0.25*10*np.pi*np.cos(10*np.pi*x)*np.sin(10*np.pi*y)-0.25*20*np.pi*np.cos(20*np.pi*x)*np.sin(20*np.pi*y)

        fside = -1.0*(Aepsx*ux + Aepsy*uy + Aeps*(uxx+uyy)) + 1.0*np.pi*np.pi*u
    elif equa_name == 'Boltzmann6':
        u = np.sin(np.pi*x)*np.sin(np.pi*y)+0.05*np.sin(10*np.pi*x)*np.sin(10*np.pi*y)+0.01*np.sin(20*np.pi*x)*np.sin(20*np.pi*y)
        Aeps = 0.5+0.25*np.cos(np.pi*x)*np.cos(np.pi*y) + 0.25*np.cos(20*np.pi*x)*np.cos(20*np.pi*y)

        ux = 1.0*np.pi*np.cos(np.pi*x)*np.sin(np.pi*y)+0.5*np.pi*np.cos(10*np.pi*x)*np.sin(10*np.pi*y)+0.2*np.pi*np.cos(20*np.pi*x)*np.sin(20*np.pi*y)
        uy = 1.0*np.pi*np.sin(np.pi*x)*np.cos(np.pi*y)+0.5*np.pi*np.sin(10*np.pi*x)*np.cos(10*np.pi*y)+0.2*np.pi*np.sin(20*np.pi*x)*np.cos(20*np.pi*y)
        uxx = -1.0*np.pi*np.pi*np.sin(np.pi*x)*np.sin(np.pi*y)-5.0*np.pi*np.pi*np.sin(10*np.pi*x)*np.sin(10*np.pi*y)-4.0*np.pi*np.pi*np.sin(20*np.pi*x)*np.sin(20*np.pi*y)
        uyy = -1.0*np.pi*np.pi*np.sin(np.pi*x)*np.sin(np.pi*y)-5.0*np.pi*np.pi*np.sin(10*np.pi*x)*np.sin(10*np.pi*y)-4.0*np.pi*np.pi*np.sin(20*np.pi*x)*np.sin(20*np.pi*y)

        Aepsx = -0.25*np.pi*np.sin(np.pi*x)*np.cos(np.pi*y)-0.25*20*np.pi*np.sin(20*np.pi*x)*np.cos(20*np.pi*y)
        Aepsy = -0.25*np.pi*np.cos(np.pi*x)*np.sin(np.pi*y)-0.25*20*np.pi*np.cos(20*np.pi*x)*np.sin(20*np.pi*y)

        fside = -1.0*(Aepsx*ux + Aepsy*uy + Aeps*(uxx+uyy)) + 1.0*np.pi*np.pi*u

    return fside


def get_infos2Boltzmann_3D(input_dim=1, out_dim=1, mesh_number=2, intervalL=0.0, intervalR=1.0, equa_name=None):
    if equa_name == 'Boltzmann0':
        # mu1= 2*np.pi
        # mu2 = 4*np.pi
        # mu3 = 8*np.pi
        # mu1 = np.pi
        # mu2 = 5 * np.pi
        # mu3 = 10 * np.pi
        mu1 = np.pi
        mu2 = 10 * np.pi
        mu3 = 20 * np.pi
        fside = lambda x, y, z: (mu1*mu1+mu2*mu2+mu3*mu3+x*x+2*y*y+3*z*z)*np.sin(mu1*x)*np.sin(mu2*y)*np.sin(mu3*z)
        A_eps = lambda x, y, z: 1.0*np.ones_like(x)
        kappa = lambda x, y, z: x*x+2*y*y+3*z*z
        utrue = lambda x, y, z: np.sin(mu1*x)*np.sin(mu2*y)*np.sin(mu3*z)
        u_00 = lambda x, y, z: np.sin(mu1*intervalL)*np.sin(mu2*y)*np.sin(mu3*z)
        u_01 = lambda x, y, z: np.sin(mu1*intervalR)*np.sin(mu2*y)*np.sin(mu3*z)
        u_10 = lambda x, y, z: np.sin(mu1*x)*np.sin(mu2*intervalL)*np.sin(mu3*z)
        u_11 = lambda x, y, z: np.sin(mu1*x)*np.sin(mu2*intervalR)*np.sin(mu3*z)
        u_20 = lambda x, y, z: np.sin(mu1*x)*np.sin(mu2*y)*np.sin(mu3*intervalL)
        u_21 = lambda x, y, z: np.sin(mu1*x)*np.sin(mu2*y)*np.sin(mu3*intervalR)

        return A_eps, kappa, fside, utrue, u_00, u_01, u_10, u_11, u_20, u_21
    if equa_name == 'Boltzmann1':
        fside = lambda x, y, z: np.ones_like(x)
        utrue = lambda x, y, z: np.sin(np.pi * x) * np.sin(5 * np.pi * y) * np.sin(10 * np.pi * z)
        A_eps = lambda x, y, z: 0.25 * (1.0 + np.cos(np.pi * x) * np.cos(10 * np.pi * y) * np.cos(20 * np.pi * z))
        kappa = lambda x, y, z: np.ones_like(x)*(np.pi)*(np.pi)
        u_00 = lambda x, y, z: np.sin(np.pi * intervalL) * np.sin(5 * np.pi * y) * np.sin(10*np.pi * z)
        u_01 = lambda x, y, z: np.sin(np.pi * intervalR) * np.sin(5 * np.pi * y) * np.sin(10*np.pi * z)
        u_10 = lambda x, y, z: np.sin(np.pi * x) * np.sin(5 * np.pi * intervalL) * np.sin(10*np.pi * z)
        u_11 = lambda x, y, z: np.sin(np.pi * x) * np.sin(5 * np.pi * intervalR) * np.sin(10*np.pi * z)
        u_20 = lambda x, y, z: np.sin(np.pi * x) * np.sin(np.pi * y) * np.sin(10 * np.pi * intervalL)
        u_21 = lambda x, y, z: np.sin(np.pi * x) * np.sin(np.pi * y) * np.sin(10 * np.pi * intervalR)
        return A_eps, kappa, fside, utrue, u_00, u_01, u_10, u_11, u_20, u_21
    if equa_name == 'Boltzmann2':
        fside = lambda x, y, z: (63 / 4) * ((np.pi) ** 2) * (
                    1.0 + np.cos(np.pi * x) * np.cos(10 * np.pi * y) * np.cos(20 * np.pi * z)) * \
                                (np.sin(np.pi * x) * np.sin(5 * np.pi * y) * np.sin(10 * np.pi * z)) + \
                                0.125 * ((np.pi) ** 2) * np.sin(np.pi * x) * np.cos(10 * np.pi * y) * np.cos(
            20 * np.pi * z) * np.cos(np.pi * x) * np.sin(5 * np.pi * y) * np.sin(10 * np.pi * z) + \
                                (25 / 4) * ((np.pi) ** 2) * np.cos(np.pi * x) * np.sin(10 * np.pi * y) * np.cos(
            20 * np.pi * z) * np.sin(np.pi * x) * np.cos(5 * np.pi * y) * np.sin(10 * np.pi * z) + \
                                25.0 * ((np.pi) ** 2) * np.cos(np.pi * x) * np.cos(10 * np.pi * y) * np.sin(
            20 * np.pi * z) * np.sin(np.pi * x) * np.sin(5 * np.pi * y) * np.cos(10 * np.pi * z) + \
                                0.5 * (np.pi * np.pi) * np.sin(np.pi * x) * np.sin(5 * np.pi * y) * np.sin(10 * np.pi * z)
        utrue = lambda x, y, z: 0.5 * np.sin(np.pi * x) * np.sin(5 * np.pi * y) * np.sin(10 * np.pi * z)
        A_eps = lambda x, y, z: 0.25 * (1.0 + np.cos(np.pi * x) * np.cos(10 * np.pi * y) * np.cos(20 * np.pi * z))
        kappa = lambda x, y, z: np.ones_like(x)*(np.pi)*(np.pi)
        u_00 = lambda x, y, z: 0.5 * np.sin(np.pi * intervalL) * np.sin(5 * np.pi * y) * np.sin(10*np.pi * z)
        u_01 = lambda x, y, z: 0.5 * np.sin(np.pi * intervalR) * np.sin(5 * np.pi * y) * np.sin(10*np.pi * z)
        u_10 = lambda x, y, z: 0.5 * np.sin(np.pi * x) * np.sin(5 * np.pi * intervalL) * np.sin(10*np.pi * z)
        u_11 = lambda x, y, z: 0.5 * np.sin(np.pi * x) * np.sin(5 * np.pi * intervalR) * np.sin(10*np.pi * z)
        u_20 = lambda x, y, z: 0.5 * np.sin(np.pi * x) * np.sin(np.pi * y) * np.sin(10 * np.pi * intervalL)
        u_21 = lambda x, y, z: 0.5 * np.sin(np.pi * x) * np.sin(np.pi * y) * np.sin(10 * np.pi * intervalR)
        return A_eps, kappa, fside, utrue, u_00, u_01, u_10, u_11, u_20, u_21
    if equa_name == 'Boltzmann3':
        fside = lambda x, y, z: np.ones_like(x)
        utrue = lambda x, y, z: np.sin(2*np.pi * x) * np.sin(10 * np.pi * y) * np.sin(2 * np.pi * z)
        A_eps = lambda x, y, z: 0.5 * (1.0 + np.cos(10*np.pi * x) * np.cos(20 * np.pi * y) * np.cos(10 * np.pi * z))
        kappa = lambda x, y, z: np.ones_like(x)*(np.pi)*(np.pi)
        u_00 = lambda x, y, z: np.sin(2*np.pi * intervalL) * np.sin(10 * np.pi * y) * np.sin(2*np.pi * z)
        u_01 = lambda x, y, z: np.sin(2*np.pi * intervalR) * np.sin(10 * np.pi * y) * np.sin(2*np.pi * z)
        u_10 = lambda x, y, z: np.sin(2*np.pi * x) * np.sin(10 * np.pi * intervalL) * np.sin(2*np.pi * z)
        u_11 = lambda x, y, z: np.sin(2*np.pi * x) * np.sin(10 * np.pi * intervalR) * np.sin(2*np.pi * z)
        u_20 = lambda x, y, z: np.sin(2*np.pi * x) * np.sin(10*np.pi * y) * np.sin(2 * np.pi * intervalL)
        u_21 = lambda x, y, z: np.sin(2*np.pi * x) * np.sin(10*np.pi * y) * np.sin(2 * np.pi * intervalR)
        return A_eps, kappa, fside, utrue, u_00, u_01, u_10, u_11, u_20, u_21
    if equa_name == 'Boltzmann4':
        fside = lambda x, y, z: np.ones_like(x)
        utrue = lambda x, y, z: np.sin(np.pi*x)*np.sin(np.pi*y)*np.sin(np.pi*z) + 0.05*np.sin(10*np.pi*x)*np.sin(20*np.pi*y)*np.sin(10*np.pi*z)
        A_eps = lambda x, y, z: 0.25 * (1.0 + np.cos(10*np.pi * x) * np.cos(20 * np.pi * y) * np.cos(10 * np.pi * z))
        kappa = lambda x, y, z: np.ones_like(x)*(np.pi)*(np.pi)
        u_00 = lambda x, y, z: np.sin(np.pi*intervalL)*np.sin(np.pi*y)*np.sin(np.pi*z) + 0.05*np.sin(10*np.pi*intervalL)*np.sin(20*np.pi*y)*np.sin(10*np.pi*z)
        u_01 = lambda x, y, z: np.sin(np.pi*intervalR)*np.sin(np.pi*y)*np.sin(np.pi*z) + 0.05*np.sin(10*np.pi*intervalR)*np.sin(20*np.pi*y)*np.sin(10*np.pi*z)
        u_10 = lambda x, y, z: np.sin(np.pi*x)*np.sin(np.pi*intervalL)*np.sin(np.pi*z) + 0.05*np.sin(10*np.pi*x)*np.sin(20*np.pi*intervalL)*np.sin(10*np.pi*z)
        u_11 = lambda x, y, z: np.sin(np.pi*x)*np.sin(np.pi*intervalR)*np.sin(np.pi*z) + 0.05*np.sin(10*np.pi*x)*np.sin(20*np.pi*intervalR)*np.sin(10*np.pi*z)
        u_20 = lambda x, y, z: np.sin(np.pi*x)*np.sin(np.pi*y)*np.sin(np.pi*intervalL) + 0.05*np.sin(10*np.pi*x)*np.sin(20*np.pi*y)*np.sin(10*np.pi*intervalL)
        u_21 = lambda x, y, z: np.sin(np.pi*x)*np.sin(np.pi*y)*np.sin(np.pi*intervalR) + 0.05*np.sin(10*np.pi*x)*np.sin(20*np.pi*y)*np.sin(10*np.pi*intervalR)
        return A_eps, kappa, fside, utrue, u_00, u_01, u_10, u_11, u_20, u_21
    if equa_name == 'Boltzmann5':
        fside = lambda x, y, z: np.ones_like(x)
        utrue = lambda x, y, z: np.sin(np.pi*x)*np.sin(np.pi*y)*np.sin(np.pi*z) + 0.05*np.sin(5*np.pi*x)*np.sin(5*np.pi*y)*np.sin(5*np.pi*z)
        A_eps = lambda x, y, z: 0.5 * (1.0 + np.cos(10.0*np.pi * x) * np.cos(10.0 * np.pi * y) * np.cos(10.0 * np.pi * z))
        kappa = lambda x, y, z: np.ones_like(x)*(np.pi)*(np.pi)
        u_00 = lambda x, y, z: np.sin(np.pi*intervalL)*np.sin(np.pi*y)*np.sin(np.pi*z) + 0.05*np.sin(5*np.pi*intervalL)*np.sin(5*np.pi*y)*np.sin(5*np.pi*z)
        u_01 = lambda x, y, z: np.sin(np.pi*intervalR)*np.sin(np.pi*y)*np.sin(np.pi*z) + 0.05*np.sin(5*np.pi*intervalR)*np.sin(5*np.pi*y)*np.sin(5*np.pi*z)
        u_10 = lambda x, y, z: np.sin(np.pi*x)*np.sin(np.pi*intervalL)*np.sin(np.pi*z) + 0.05*np.sin(5*np.pi*x)*np.sin(5*np.pi*intervalL)*np.sin(5*np.pi*z)
        u_11 = lambda x, y, z: np.sin(np.pi*x)*np.sin(np.pi*intervalR)*np.sin(np.pi*z) + 0.05*np.sin(5*np.pi*x)*np.sin(5*np.pi*intervalR)*np.sin(5*np.pi*z)
        u_20 = lambda x, y, z: np.sin(np.pi*x)*np.sin(np.pi*y)*np.sin(np.pi*intervalL) + 0.05*np.sin(5*np.pi*x)*np.sin(5*np.pi*y)*np.sin(5*np.pi*intervalL)
        u_21 = lambda x, y, z: np.sin(np.pi*x)*np.sin(np.pi*y)*np.sin(np.pi*intervalR) + 0.05*np.sin(5*np.pi*x)*np.sin(5*np.pi*y)*np.sin(5*np.pi*intervalR)
        return A_eps, kappa, fside, utrue, u_00, u_01, u_10, u_11, u_20, u_21
    if equa_name == 'Boltzmann6':
        fside = lambda x, y, z: np.ones_like(x)
        utrue = lambda x, y, z: np.sin(np.pi*x)*np.sin(np.pi*y)*np.sin(np.pi*z)
        A_eps = lambda x, y, z: 0.5 * (1.0 + np.cos(10.0*np.pi * x) * np.cos(10.0 * np.pi * y) * np.cos(10.0 * np.pi * z))
        kappa = lambda x, y, z: np.ones_like(x)*(np.pi)*(np.pi)
        u_00 = lambda x, y, z: np.sin(np.pi*intervalL)*np.sin(np.pi*y)*np.sin(np.pi*z)
        u_01 = lambda x, y, z: np.sin(np.pi*intervalR)*np.sin(np.pi*y)*np.sin(np.pi*z)
        u_10 = lambda x, y, z: np.sin(np.pi*x)*np.sin(np.pi*intervalL)*np.sin(np.pi*z)
        u_11 = lambda x, y, z: np.sin(np.pi*x)*np.sin(np.pi*intervalR)*np.sin(np.pi*z)
        u_20 = lambda x, y, z: np.sin(np.pi*x)*np.sin(np.pi*y)*np.sin(np.pi*intervalL)
        u_21 = lambda x, y, z: np.sin(np.pi*x)*np.sin(np.pi*y)*np.sin(np.pi*intervalR)
        return A_eps, kappa, fside, utrue, u_00, u_01, u_10, u_11, u_20, u_21
    if equa_name == 'Boltzmann7':
        fside = lambda x, y, z: np.ones_like(x)
        utrue = lambda x, y, z: np.sin(1.0*np.pi*x)*np.sin(5*np.pi*y)*np.sin(1.0*np.pi*z)
        A_eps = lambda x, y, z: 0.5 * (2.0 + np.cos(20.0*np.pi * x) * np.cos(20.0 * np.pi * y) * np.cos(20.0 * np.pi * z))
        kappa = lambda x, y, z: np.ones_like(x)*(np.pi)*(np.pi)
        u_00 = lambda x, y, z: 1.0*np.sin(np.pi*intervalL)*np.sin(5.0*np.pi*y)*np.sin(np.pi*z)
        u_01 = lambda x, y, z: 1.0*np.sin(np.pi*intervalR)*np.sin(5.0*np.pi*y)*np.sin(np.pi*z)
        u_10 = lambda x, y, z: 1.0*np.sin(np.pi*x)*np.sin(5.0*np.pi*intervalL)*np.sin(np.pi*z)
        u_11 = lambda x, y, z: 1.0*np.sin(np.pi*x)*np.sin(5.0*np.pi*intervalR)*np.sin(np.pi*z)
        u_20 = lambda x, y, z: 1.0*np.sin(np.pi*x)*np.sin(np.pi*y)*np.sin(5.0*np.pi*intervalL)
        u_21 = lambda x, y, z: 1.0*np.sin(np.pi*x)*np.sin(np.pi*y)*np.sin(5.0*np.pi*intervalR)
        return A_eps, kappa, fside, utrue, u_00, u_01, u_10, u_11, u_20, u_21
    if equa_name == 'Boltzmann8':
        fside = lambda x, y, z: np.ones_like(x)
        utrue = lambda x, y, z: np.sin(np.pi*x)*np.sin(np.pi*y)*np.sin(np.pi*z)
        A_eps = lambda x, y, z: 0.5 * (2.0 + np.cos(20.0*np.pi * x) * np.cos(20.0 * np.pi * y) * np.cos(20.0 * np.pi * z))
        kappa = lambda x, y, z: np.ones_like(x)*(np.pi)*(np.pi)
        u_00 = lambda x, y, z: np.sin(np.pi*intervalL)*np.sin(np.pi*y)*np.sin(np.pi*z)
        u_01 = lambda x, y, z: np.sin(np.pi*intervalR)*np.sin(np.pi*y)*np.sin(np.pi*z)
        u_10 = lambda x, y, z: np.sin(np.pi*x)*np.sin(np.pi*intervalL)*np.sin(np.pi*z)
        u_11 = lambda x, y, z: np.sin(np.pi*x)*np.sin(np.pi*intervalR)*np.sin(np.pi*z)
        u_20 = lambda x, y, z: np.sin(np.pi*x)*np.sin(np.pi*y)*np.sin(np.pi*intervalL)
        u_21 = lambda x, y, z: np.sin(np.pi*x)*np.sin(np.pi*y)*np.sin(np.pi*intervalR)
        return A_eps, kappa, fside, utrue, u_00, u_01, u_10, u_11, u_20, u_21
    if equa_name == 'Boltzmann9':
        fside = lambda x, y, z: np.ones_like(x)
        utrue = lambda x, y, z: np.sin(np.pi*x)*np.sin(np.pi*y)*np.sin(np.pi*z)
        A_eps = lambda x, y, z: 0.5 * (1.0 + np.cos(20.0*np.pi * x) * np.cos(20.0 * np.pi * y) * np.cos(20.0 * np.pi * z))
        kappa = lambda x, y, z: np.ones_like(x)*(np.pi)*(np.pi)
        u_00 = lambda x, y, z: np.sin(np.pi*intervalL)*np.sin(np.pi*y)*np.sin(np.pi*z)
        u_01 = lambda x, y, z: np.sin(np.pi*intervalR)*np.sin(np.pi*y)*np.sin(np.pi*z)
        u_10 = lambda x, y, z: np.sin(np.pi*x)*np.sin(np.pi*intervalL)*np.sin(np.pi*z)
        u_11 = lambda x, y, z: np.sin(np.pi*x)*np.sin(np.pi*intervalR)*np.sin(np.pi*z)
        u_20 = lambda x, y, z: np.sin(np.pi*x)*np.sin(np.pi*y)*np.sin(np.pi*intervalL)
        u_21 = lambda x, y, z: np.sin(np.pi*x)*np.sin(np.pi*y)*np.sin(np.pi*intervalR)
        return A_eps, kappa, fside, utrue, u_00, u_01, u_10, u_11, u_20, u_21
    if equa_name == 'Boltzmann10':
        fside = lambda x, y, z: np.ones_like(x)
        utrue = lambda x, y, z: np.sin(np.pi*x)*np.sin(np.pi*y)*np.sin(np.pi*z)
        A_eps = lambda x, y, z: 0.25 * (2.0 + np.cos(np.pi*x) * np.cos(np.pi*y) * np.cos(np.pi*z) + np.cos(20.0*np.pi*x) * np.cos(20.0*np.pi*y) * np.cos(20.0*np.pi*z))
        kappa = lambda x, y, z: np.ones_like(x)*(np.pi)*(np.pi)
        u_00 = lambda x, y, z: np.sin(np.pi*intervalL)*np.sin(np.pi*y)*np.sin(np.pi*z)
        u_01 = lambda x, y, z: np.sin(np.pi*intervalR)*np.sin(np.pi*y)*np.sin(np.pi*z)
        u_10 = lambda x, y, z: np.sin(np.pi*x)*np.sin(np.pi*intervalL)*np.sin(np.pi*z)
        u_11 = lambda x, y, z: np.sin(np.pi*x)*np.sin(np.pi*intervalR)*np.sin(np.pi*z)
        u_20 = lambda x, y, z: np.sin(np.pi*x)*np.sin(np.pi*y)*np.sin(np.pi*intervalL)
        u_21 = lambda x, y, z: np.sin(np.pi*x)*np.sin(np.pi*y)*np.sin(np.pi*intervalR)
        return A_eps, kappa, fside, utrue, u_00, u_01, u_10, u_11, u_20, u_21


def get_force2Boltzmann3D_E05(x=None, y=None, z=None):
    u = np.sin(np.pi * x) * np.sin(np.pi * y) * np.sin(np.pi * z) + 0.05 * np.sin(5.0 * np.pi * x) * np.sin(
        5.0 * np.pi * y) * np.sin(5.0 * np.pi * z)
    Aeps = 0.25 * (1.0 + np.cos(10.0 * np.pi * x) * np.cos(10.0 * np.pi * y) * np.cos(10.0 * np.pi * z))
    ux = np.pi * np.cos(np.pi * x) * np.sin(np.pi * y) * np.sin(np.pi * z) + 0.25 * np.pi * np.cos(
        5 * np.pi * x) * np.sin(5 * np.pi * y) * np.sin(5 * np.pi * z)
    uy = np.pi * np.sin(np.pi * x) * np.cos(np.pi * y) * np.sin(np.pi * z) + 0.25 * np.pi * np.sin(
        5 * np.pi * x) * np.cos(5 * np.pi * y) * np.sin(5 * np.pi * z)
    uz = np.pi * np.sin(np.pi * x) * np.sin(np.pi * y) * np.cos(np.pi * z) + 0.25 * np.pi * np.sin(
        5 * np.pi * x) * np.sin(5 * np.pi * y) * np.cos(5 * np.pi * z)

    uxx = -1.0 * np.pi * np.pi * np.sin(np.pi * x) * np.sin(np.pi * y) * np.sin(
        np.pi * z) - 1.25 * np.pi * np.pi * np.sin(5 * np.pi * x) * np.sin(5 * np.pi * y) * np.sin(5 * np.pi * z)
    uyy = -1.0 * np.pi * np.pi * np.sin(np.pi * x) * np.sin(np.pi * y) * np.sin(
        np.pi * z) - 1.25 * np.pi * np.pi * np.sin(5 * np.pi * x) * np.sin(5 * np.pi * y) * np.sin(5 * np.pi * z)
    uzz = -1.0 * np.pi * np.pi * np.sin(np.pi * x) * np.sin(np.pi * y) * np.sin(
        np.pi * z) - 1.25 * np.pi * np.pi * np.sin(5 * np.pi * x) * np.sin(5 * np.pi * y) * np.sin(5 * np.pi * z)

    Aepsx = -0.25 * 10.0 * np.pi * np.sin(10.0 * np.pi * x) * np.cos(10.0 * np.pi * y) * np.cos(10.0 * np.pi * z)
    Aepsy = -0.25 * 10.0 * np.pi * np.cos(10.0 * np.pi * x) * np.sin(10.0 * np.pi * y) * np.cos(10.0 * np.pi * z)
    Aepsz = -0.25 * 10.0 * np.pi * np.cos(10.0 * np.pi * x) * np.cos(10.0 * np.pi * y) * np.sin(10.0 * np.pi * z)

    fside = -1.0 * (Aepsx * ux + Aepsy * uy + Aepsz * uz + Aeps * (uxx + uyy + uzz)) + (np.pi) * (np.pi) * u
    return fside


def get_force2Boltzmann3D(x=None, y=None, z=None, equa_name=None):
    if equa_name == 'Boltzmann1':
        u = np.sin(np.pi * x) * np.sin(5 * np.pi * y) * np.sin(10 * np.pi * z)
        Aeps = 0.25 * (1.0 + np.cos(np.pi * x) * np.cos(10 * np.pi * y) * np.cos(20 * np.pi * z))
        ux = 1.0*np.pi*np.cos(np.pi * x) * np.sin(5 * np.pi * y) * np.sin(10 * np.pi * z)
        uy = 5.0*np.pi*np.sin(np.pi * x) * np.cos(5 * np.pi * y) * np.sin(10 * np.pi * z)
        uz = 10.0*np.pi*np.sin(np.pi * x) * np.sin(5 * np.pi * y) * np.cos(10 * np.pi * z)

        uxx = -1.0*np.pi*np.pi*np.sin(np.pi * x) * np.sin(5 * np.pi * y) * np.sin(10 * np.pi * z)
        uyy = -25.0*np.pi*np.pi*np.sin(np.pi * x) * np.sin(5 * np.pi * y) * np.sin(10 * np.pi * z)
        uzz = -100.0*np.pi*np.pi*np.sin(np.pi * x) * np.sin(5 * np.pi * y) * np.sin(10 * np.pi * z)

        Aepsx = -0.25*np.pi*np.sin(np.pi * x) * np.cos(10 * np.pi * y) * np.cos(20 * np.pi * z)
        Aepsy = -0.25*10*np.pi*np.cos(np.pi * x) * np.sin(10 * np.pi * y) * np.cos(20 * np.pi * z)
        Aepsz = -0.25*20*np.pi*np.cos(np.pi * x) * np.cos(10 * np.pi * y) * np.sin(20 * np.pi * z)

        fside = -1.0*(Aepsx*ux + Aepsy*uy + Aepsz*uz + Aeps*(uxx+uyy+uzz)) + (np.pi)*(np.pi)*u
    elif equa_name == 'Boltzmann3':
        u = np.sin(2*np.pi * x) * np.sin(10 * np.pi * y) * np.sin(2 * np.pi * z)
        Aeps = 0.5 * (1.0 + np.cos(10*np.pi * x) * np.cos(20 * np.pi * y) * np.cos(10 * np.pi * z))
        ux = 2*np.pi*np.cos(2*np.pi * x) * np.sin(10 * np.pi * y) * np.sin(2 * np.pi * z)
        uy = 10*np.pi*np.sin(2*np.pi * x) * np.cos(10 * np.pi * y) * np.sin(2 * np.pi * z)
        uz = 2*np.pi*np.sin(2*np.pi * x) * np.sin(10 * np.pi * y) * np.cos(2 * np.pi * z)

        uxx = -4.0*np.pi*np.pi*np.sin(2*np.pi * x) * np.sin(10 * np.pi * y) * np.sin(2 * np.pi * z)
        uyy = -100*np.pi*np.pi*np.sin(2*np.pi * x) * np.sin(10 * np.pi * y) * np.sin(2 * np.pi * z)
        uzz = -4.0*np.pi*np.pi*np.sin(2*np.pi * x) * np.sin(10 * np.pi * y) * np.sin(2 * np.pi * z)

        Aepsx = -0.5*10*np.pi*np.sin(10*np.pi * x) * np.cos(20 * np.pi * y) * np.cos(10 * np.pi * z)
        Aepsy = -0.5*20*np.pi*np.cos(10*np.pi * x) * np.sin(20 * np.pi * y) * np.cos(10 * np.pi * z)
        Aepsz = -0.5*10*np.pi*np.cos(10*np.pi * x) * np.cos(20 * np.pi * y) * np.sin(10 * np.pi * z)

        fside = -1.0*(Aepsx*ux + Aepsy*uy + Aepsz*uz + Aeps*(uxx+uyy+uzz)) + (np.pi)*(np.pi)*u
    elif equa_name == 'Boltzmann4':
        u = np.sin(np.pi*x)*np.sin(np.pi*y)*np.sin(np.pi*z) + 0.05*np.sin(10*np.pi*x)*np.sin(20*np.pi*y)*np.sin(10*np.pi*z)
        Aeps = 0.25 * (1.0 + np.cos(10*np.pi * x) * np.cos(20 * np.pi * y) * np.cos(10 * np.pi * z))
        ux = 1.0*np.pi*np.cos(np.pi*x)*np.sin(np.pi*y)*np.sin(np.pi*z) + 0.5*np.pi*np.cos(10*np.pi*x)*np.sin(20*np.pi*y)*np.sin(10*np.pi*z)
        uy = 1.0*np.pi*np.sin(np.pi*x)*np.cos(np.pi*y)*np.sin(np.pi*z) + 1.0*np.pi*np.sin(10*np.pi*x)*np.cos(20*np.pi*y)*np.sin(10*np.pi*z)
        uz = 1.0*np.pi*np.sin(np.pi*x)*np.sin(np.pi*y)*np.cos(np.pi*z) + 0.5*np.pi*np.sin(10*np.pi*x)*np.sin(20*np.pi*y)*np.cos(10*np.pi*z)

        uxx = -1.0*np.pi*np.pi*np.sin(np.pi*x)*np.sin(np.pi*y)*np.sin(np.pi*z) - 5.0*np.pi*np.pi*np.sin(10*np.pi*x)*np.sin(20*np.pi*y)*np.sin(10*np.pi*z)
        uyy = -1.0*np.pi*np.pi*np.sin(np.pi*x)*np.sin(np.pi*y)*np.sin(np.pi*z) - 20.0*np.pi*np.pi*np.sin(10*np.pi*x)*np.sin(20*np.pi*y)*np.sin(10*np.pi*z)
        uzz = -1.0*np.pi*np.pi*np.sin(np.pi*x)*np.sin(np.pi*y)*np.sin(np.pi*z) - 5.0*np.pi*np.pi*np.sin(10*np.pi*x)*np.sin(20*np.pi*y)*np.sin(10*np.pi*z)

        Aepsx = -0.25*10*np.pi*np.sin(10*np.pi * x) * np.cos(20 * np.pi * y) * np.cos(10 * np.pi * z)
        Aepsy = -0.25*20*np.pi*np.cos(10*np.pi * x) * np.sin(20 * np.pi * y) * np.cos(10 * np.pi * z)
        Aepsz = -0.25*10*np.pi*np.cos(10*np.pi * x) * np.cos(20 * np.pi * y) * np.sin(10 * np.pi * z)

        fside = -1.0*(Aepsx*ux + Aepsy*uy + Aepsz*uz + Aeps*(uxx+uyy+uzz)) + (np.pi)*(np.pi)*u
    elif equa_name == 'Boltzmann5':
        u = np.sin(np.pi*x)*np.sin(np.pi*y)*np.sin(np.pi*z) + 0.05*np.sin(5*np.pi*x)*np.sin(5*np.pi*y)*np.sin(5*np.pi*z)
        Aeps = 0.25 * (2.0 + np.cos(10.0*np.pi * x) * np.cos(10.0 * np.pi * y) * np.cos(10.0 * np.pi * z))
        ux = np.pi*np.cos(np.pi*x)*np.sin(np.pi*y)*np.sin(np.pi*z) + 0.25*np.pi*np.cos(5*np.pi*x)*np.sin(5*np.pi*y)*np.sin(5*np.pi*z)
        uy = np.pi*np.sin(np.pi*x)*np.cos(np.pi*y)*np.sin(np.pi*z) + 0.25*np.pi*np.sin(5*np.pi*x)*np.cos(5*np.pi*y)*np.sin(5*np.pi*z)
        uz = np.pi*np.sin(np.pi*x)*np.sin(np.pi*y)*np.cos(np.pi*z) + 0.25*np.pi*np.sin(5*np.pi*x)*np.sin(5*np.pi*y)*np.cos(5*np.pi*z)

        uxx = -1.0*np.pi*np.pi*np.sin(np.pi*x)*np.sin(np.pi*y)*np.sin(np.pi*z) - 1.25*np.pi*np.pi*np.sin(5*np.pi*x)*np.sin(5*np.pi*y)*np.sin(5*np.pi*z)
        uyy = -1.0*np.pi*np.pi*np.sin(np.pi*x)*np.sin(np.pi*y)*np.sin(np.pi*z) - 1.25*np.pi*np.pi*np.sin(5*np.pi*x)*np.sin(5*np.pi*y)*np.sin(5*np.pi*z)
        uzz = -1.0*np.pi*np.pi*np.sin(np.pi*x)*np.sin(np.pi*y)*np.sin(np.pi*z) - 1.25*np.pi*np.pi*np.sin(5*np.pi*x)*np.sin(5*np.pi*y)*np.sin(5*np.pi*z)

        Aepsx = -0.25*10.0*np.pi*np.sin(10.0*np.pi * x) * np.cos(10.0 * np.pi * y) * np.cos(10.0 * np.pi * z)
        Aepsy = -0.25*10.0*np.pi*np.cos(10.0*np.pi * x) * np.sin(10.0 * np.pi * y) * np.cos(10.0 * np.pi * z)
        Aepsz = -0.25*10.0*np.pi*np.cos(10.0*np.pi * x) * np.cos(10.0 * np.pi * y) * np.sin(10.0 * np.pi * z)

        fside = -1.0*(Aepsx*ux + Aepsy*uy + Aepsz*uz + Aeps*(uxx+uyy+uzz)) + (np.pi)*(np.pi)*u
    elif equa_name == 'Boltzmann6':
        u = np.sin(np.pi*x)*np.sin(np.pi*y)*np.sin(np.pi*z)
        Aeps = 0.5 * (1.0 + np.cos(10 * np.pi * x) * np.cos(10 * np.pi * y) * np.cos(10 * np.pi * z))
        ux = np.pi*np.cos(np.pi*x)*np.sin(np.pi*y)*np.sin(np.pi*z)
        uy = np.pi*np.sin(np.pi*x)*np.cos(np.pi*y)*np.sin(np.pi*z)
        uz = np.pi*np.sin(np.pi*x)*np.sin(np.pi*y)*np.cos(np.pi*z)

        uxx = -1.0*np.pi*np.pi*np.sin(np.pi*x)*np.sin(np.pi*y)*np.sin(np.pi*z)
        uyy = -1.0*np.pi*np.pi*np.sin(np.pi*x)*np.sin(np.pi*y)*np.sin(np.pi*z)
        uzz = -1.0*np.pi*np.pi*np.sin(np.pi*x)*np.sin(np.pi*y)*np.sin(np.pi*z)

        Aepsx = -5.0*np.pi*np.sin(10*np.pi * x) * np.cos(10 * np.pi * y) * np.cos(10*np.pi * z)
        Aepsy = -5.0*np.pi*np.cos(10*np.pi * x) * np.sin(10 * np.pi * y) * np.cos(10*np.pi * z)
        Aepsz = -5.0*np.pi*np.cos(10*np.pi * x) * np.cos(10 * np.pi * y) * np.sin(10*np.pi * z)

        fside = -1.0*(Aepsx*ux + Aepsy*uy + Aepsz*uz + Aeps*(uxx+uyy+uzz)) + np.pi*np.pi*u
    elif equa_name == 'Boltzmann7':
        u = np.sin(np.pi*x)*np.sin(5.0*np.pi*y)*np.sin(np.pi*z)
        Aeps = 0.5 * (2.0 + np.cos(20.0*np.pi*x) * np.cos(20.0*np.pi*y) * np.cos(20.0*np.pi*z))
        ux = 1.0*np.pi*np.cos(np.pi*x)*np.sin(5.0*np.pi*y)*np.sin(np.pi*z)
        uy = 5.0*np.pi*np.sin(np.pi*x)*np.cos(5.0*np.pi*y)*np.sin(np.pi*z)
        uz = 1.0*np.pi*np.sin(np.pi*x)*np.sin(5.0*np.pi*y)*np.cos(np.pi*z)

        uxx = -1.0*np.pi*np.pi*np.sin(np.pi*x)*np.sin(5.0*np.pi*y)*np.sin(np.pi*z)
        uyy = -25.0*np.pi*np.pi*np.sin(np.pi*x)*np.sin(5.0*np.pi*y)*np.sin(np.pi*z)
        uzz = -1.0*np.pi*np.pi*np.sin(np.pi*x)*np.sin(5.0*np.pi*y)*np.sin(np.pi*z)

        Aepsx = -0.5*20.0*np.pi*np.sin(20.0*np.pi * x) * np.cos(20.0 * np.pi * y) * np.cos(20.0 * np.pi * z)
        Aepsy = -0.5*20.0*np.pi*np.cos(20.0*np.pi * x) * np.sin(20.0 * np.pi * y) * np.cos(20.0 * np.pi * z)
        Aepsz = -0.5*20.0*np.pi*np.cos(20.0*np.pi * x) * np.cos(20.0 * np.pi * y) * np.sin(20.0 * np.pi * z)

        fside = -1.0*(Aepsx*ux + Aepsy*uy + Aepsz*uz + Aeps*(uxx+uyy+uzz)) + (np.pi)*(np.pi)*u
    elif equa_name == 'Boltzmann8':
        u = np.sin(np.pi*x)*np.sin(np.pi*y)*np.sin(np.pi*z)
        Aeps = 0.5 * (2.0 + np.cos(20.0*np.pi*x) * np.cos(20.0*np.pi*y) * np.cos(20.0*np.pi*z))
        ux = np.pi*np.cos(np.pi*x)*np.sin(np.pi*y)*np.sin(np.pi*z)
        uy = np.pi*np.sin(np.pi*x)*np.cos(np.pi*y)*np.sin(np.pi*z)
        uz = np.pi*np.sin(np.pi*x)*np.sin(np.pi*y)*np.cos(np.pi*z)

        uxx = -1.0*np.pi*np.pi*np.sin(np.pi*x)*np.sin(np.pi*y)*np.sin(np.pi*z)
        uyy = -1.0*np.pi*np.pi*np.sin(np.pi*x)*np.sin(np.pi*y)*np.sin(np.pi*z)
        uzz = -1.0*np.pi*np.pi*np.sin(np.pi*x)*np.sin(np.pi*y)*np.sin(np.pi*z)

        Aepsx = -10.0*np.pi*np.sin(20.0*np.pi * x) * np.cos(20.0 * np.pi * y) * np.cos(20.0 * np.pi * z)
        Aepsy = -10.0*np.pi*np.cos(20.0*np.pi * x) * np.sin(20.0 * np.pi * y) * np.cos(20.0 * np.pi * z)
        Aepsz = -10.0*np.pi*np.cos(20.0*np.pi * x) * np.cos(20.0 * np.pi * y) * np.sin(20.0 * np.pi * z)

        fside = -1.0*(Aepsx*ux + Aepsy*uy + Aepsz*uz + Aeps*(uxx+uyy+uzz)) + (np.pi)*(np.pi)*u
    elif equa_name == 'Boltzmann9':
        u = np.sin(np.pi*x)*np.sin(np.pi*y)*np.sin(np.pi*z)
        Aeps = 0.5 * (1.0 + np.cos(20.0*np.pi*x) * np.cos(20.0*np.pi*y) * np.cos(20.0*np.pi*z))
        ux = np.pi*np.cos(np.pi*x)*np.sin(np.pi*y)*np.sin(np.pi*z)
        uy = np.pi*np.sin(np.pi*x)*np.cos(np.pi*y)*np.sin(np.pi*z)
        uz = np.pi*np.sin(np.pi*x)*np.sin(np.pi*y)*np.cos(np.pi*z)

        uxx = -1.0*np.pi*np.pi*np.sin(np.pi*x)*np.sin(np.pi*y)*np.sin(np.pi*z)
        uyy = -1.0*np.pi*np.pi*np.sin(np.pi*x)*np.sin(np.pi*y)*np.sin(np.pi*z)
        uzz = -1.0*np.pi*np.pi*np.sin(np.pi*x)*np.sin(np.pi*y)*np.sin(np.pi*z)

        Aepsx = -10.0*np.pi*np.sin(20.0*np.pi * x) * np.cos(20.0 * np.pi * y) * np.cos(20.0 * np.pi * z)
        Aepsy = -10.0*np.pi*np.cos(20.0*np.pi * x) * np.sin(20.0 * np.pi * y) * np.cos(20.0 * np.pi * z)
        Aepsz = -10.0*np.pi*np.cos(20.0*np.pi * x) * np.cos(20.0 * np.pi * y) * np.sin(20.0 * np.pi * z)

        fside = -1.0*(Aepsx*ux + Aepsy*uy + Aepsz*uz + Aeps*(uxx+uyy+uzz)) + (np.pi)*(np.pi)*u
    elif equa_name == 'Boltzmann10':
        u = np.sin(np.pi * x) * np.sin(np.pi * y) * np.sin(np.pi * z)
        Aeps = 0.25 * (2.0 + np.cos(np.pi*x)*np.cos(np.pi*y)*np.cos(np.pi*z) + np.cos(20*np.pi*x)*np.cos(20*np.pi*y)*np.cos(20*np.pi*z))
        ux = np.pi * np.cos(np.pi * x) * np.sin(np.pi * y) * np.sin(np.pi * z)
        uy = np.pi * np.sin(np.pi * x) * np.cos(np.pi * y) * np.sin(np.pi * z)
        uz = np.pi * np.sin(np.pi * x) * np.sin(np.pi * y) * np.cos(np.pi * z)

        uxx = -1.0 * np.pi * np.pi * np.sin(np.pi * x) * np.sin(np.pi * y) * np.sin(np.pi * z)
        uyy = -1.0 * np.pi * np.pi * np.sin(np.pi * x) * np.sin(np.pi * y) * np.sin(np.pi * z)
        uzz = -1.0 * np.pi * np.pi * np.sin(np.pi * x) * np.sin(np.pi * y) * np.sin(np.pi * z)

        Aepsx = -1.0*np.pi*np.sin(np.pi*x)*np.cos(np.pi*y)*np.cos(np.pi*z) - 5.0*np.pi*np.sin(20*np.pi*x)*np.cos(20*np.pi*y)*np.cos(20*np.pi*z)
        Aepsy = -1.0*np.pi*np.cos(np.pi*x)*np.sin(np.pi*y)*np.cos(np.pi*z) - 5.0*np.pi*np.cos(20*np.pi*x)*np.sin(20*np.pi*y)*np.cos(20*np.pi*z)
        Aepsz = -1.0*np.pi*np.cos(np.pi*x)*np.cos(np.pi*y)*np.sin(np.pi*z) - 5.0*np.pi*np.cos(20*np.pi*x)*np.cos(20*np.pi*y)*np.sin(20*np.pi*z)

        fside = -1.0 * (Aepsx * ux + Aepsy * uy + Aepsz * uz + Aeps * (uxx + uyy + uzz)) + (np.pi) * (np.pi) * u

    return fside


def get_infos2Boltzmann_5D(input_dim=1, out_dim=1, mesh_number=2, intervalL=0.0, intervalR=1.0, equa_name=None):
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