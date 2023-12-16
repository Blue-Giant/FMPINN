function V = test3d_ms2f(XX,YY,ZZ,EPS)
%     eps = 0.1;
    Aeps = 2.0+sin(pi*XX/EPS)*sin(pi*YY/EPS)*sin(pi*ZZ/EPS);
    dAeps2dx = (pi/EPS)*cos(pi*XX/EPS)*sin(pi*YY/EPS)*sin(pi*ZZ/EPS);
    dAeps2dy = (pi/EPS)*sin(pi*XX/EPS)*cos(pi*YY/EPS)*sin(pi*ZZ/EPS);
    dAeps2dz = (pi/EPS)*sin(pi*XX/EPS)*sin(pi*YY/EPS)*cos(pi*ZZ/EPS);
%     uexact = sin(pi*XX)*sin(pi*YY);
    u_x = pi*cos(pi*XX)*sin(pi*YY)*sin(pi*ZZ);
    u_y = pi*sin(pi*XX)*cos(pi*YY)*sin(pi*ZZ);
    u_z = pi*sin(pi*XX)*sin(pi*YY)*cos(pi*ZZ);
    uxx = -pi^2*sin(pi*XX)*sin(pi*YY)*sin(pi*ZZ);
    uyy = -pi^2*sin(pi*XX)*sin(pi*YY)*sin(pi*ZZ);
    uzz = -pi^2*sin(pi*XX)*sin(pi*YY)*sin(pi*ZZ);
    V = dAeps2dx*u_x+dAeps2dy*u_y+dAeps2dz*u_z+Aeps*(uxx+uyy+uzz);
end