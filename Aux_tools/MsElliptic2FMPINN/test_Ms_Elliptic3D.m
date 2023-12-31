%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%       div(a，grad U) = f(x,y) (x,y)in[0,1]X[0,1]
%%%             u = 0 on boundary.
%%%          a = 2+sin(pi*x/eps)*sin(pi*y/eps)*sin(pi*z/eps)
%%%
%%%   dv(a，grad u) = a_x，u_x + a_y，u_y + a_z，u_z + a，delta u 
%%% u_x = 0.5*[u(i+1,j) - u(i-1,j)]; u_y = 0.5*[u(i,j+1) - u(i,j-1)]
clc
clear all
close all

eps = 0.1;
Aeps = @(XX,YY,ZZ)  2.0+sin(pi*XX/eps)*sin(pi*YY/eps)*sin(pi*ZZ/eps);
dAeps2dx = @(XX,YY,ZZ)  (pi/eps)*cos(pi*XX/eps)*sin(pi*YY/eps)*sin(pi*ZZ/eps);
dAeps2dy = @(XX,YY,ZZ)  (pi/eps)*sin(pi*XX/eps)*cos(pi*YY/eps)*sin(pi*ZZ/eps);
dAeps2dz = @(XX,YY,ZZ)  (pi/eps)*sin(pi*XX/eps)*sin(pi*YY/eps)*cos(pi*ZZ/eps);

N = 32;
% N = 64;
xl = 0; xr = 1;
yl = 0; yr = 1;
zl = 0; zr = 1;
h = (xr-xl)/N;

x = xl:h:xr;
y = yl:h:yr;
z = zl:h:zr;

u0 = zeros(N+1, N+1, N+1);
for j=1:N+1
    for k=1:N+1
        u0(1,j,k) = 0;
        u0(N+1,j,k) = 0;
    end
end

for i=1:N+1
    for k=1:N+1
        u0(i,1,k) = 0;
        u0(i,N+1,k) = 0;
    end
end

for i=1:N+1
    for j=1:N+1
        u0(i,j,1) = 0;
        u0(i,k,N+1) = 0;
    end
end

u = u0;

error = 100;
tol = 10^-8;
IT=0;

while error > tol
    for i=2:N
        for j=2:N
            for k=2:N
                Ax = dAeps2dx(x(i),y(j),z(k))/Aeps(x(i),y(j),z(k));
                Ay = dAeps2dy(x(i),y(j),z(k))/Aeps(x(i),y(j),z(k));
                Az = dAeps2dz(x(i),y(j),z(k))/Aeps(x(i),y(j),z(k));
                fa = test3d_ms2f(x(i),y(j),z(k),eps)/Aeps(x(i),y(j),z(k));
                temp =Ax*0.5*h*(u0(i+1,j,k)-u0(i-1,j,k))+Ay*0.5*h*(u0(i,j+1,k)-u0(i,j-1,k))+Az*0.5*h*(u0(i,j,k+1)-u0(i,j, k-1))+...
                    u0(i-1,j,k)+u0(i+1,j,k)+u0(i,j-1,k)+u0(i,j+1,k)+u0(i,j,k-1)+u0(i,j,k+1) - h^2*fa;
                u(i,j,k) =temp/6.0;
            end
        end
    end
    error = max(max(max(abs(u-u0))));
    u0 = u;
    IT = IT+1;
end

u_slice = reshape(u(10,:,:),N+1,N+1);
figure(1)
surf(y,z,u_slice);
title('Approx'); 

Uexact = @(XX,YY, ZZ) sin(pi*XX).*sin(pi*YY).*sin(pi*ZZ);
ue = zeros(N+1);
for i=1:N+1
    for j=1:N+1
        for k=1:N+1
        ue(i,j, k) =Uexact(x(i),y(j),z(k));
        end
    end
end

uexact_slice = reshape(ue(10,:,:),N+1,N+1);
figure(2)
surf(y,z,uexact_slice);
title('Exact');

abserr = abs(u_slice-uexact_slice);
figure(3)
surf(y,z,abserr);
title('Abs Err');

max_err = max(max(max(abs(u-ue))))

if N==32
    save('uref32',u)
elseif N==64
    save('uref64',u)
elseif N==100
    save('uref100',u)
end