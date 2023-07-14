function V = test_ms2f(XX,YY, EPS)
%     eps = 0.1;
    Aeps = 2.0+sin(pi*XX/EPS)*sin(pi*YY/EPS);
    dAeps2dx = (pi/EPS)*cos(pi*XX/EPS)*sin(pi*YY/EPS);
    dAeps2dy = (pi/EPS)*sin(pi*XX/EPS)*cos(pi*YY/EPS);
%     uexact = sin(pi*XX)*sin(pi*YY);
    u_x = pi*cos(pi*XX)*sin(pi*YY);
    u_y = pi*sin(pi*XX)*cos(pi*YY);
    uxx = -pi^2*sin(pi*XX)*sin(pi*YY);
    uyy = -pi^2*sin(pi*XX)*sin(pi*YY);
    V = dAeps2dx*u_x+dAeps2dy*u_y+Aeps*(uxx+uyy);
end