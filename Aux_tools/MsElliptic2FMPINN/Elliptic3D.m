%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%     u_xx+u_yy + u_yy = f(x,y,z) (x,y,z) in[0,1]x[0,1]x[0,1]
%%%            u = 0 on boundary.
%%%     u_exact = sin(pi*x)*sin(pi*y)*sin(pi*z)
%%%     u_xx = -pi*pi*sin(pi*x)*sin(pi*y)*sin(pi*z)
%%%     u_yy = -pi*pi*sin(pi*x)*sin(pi*y)*sin(pi*z)
%%%     u_zz = -pi*pi*sin(pi*x)*sin(pi*y)*sin(pi*z)
%%% u_x = 0.5*[u(i+1,j, k) - u(i-1,j, k)]; u_y = 0.5*[u(i,j+1, k) - u(i,j-1,k)]

clear all
close all
clc

f = @(XX,YY, ZZ)  -3.0*pi*pi*sin(pi*XX)*sin(pi*YY)*sin(pi*ZZ);

N = 32;
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
                temp = u0(i-1,j, k)+u0(i+1,j, k) + u0(i,j-1, k)+u0(i,j+1, k) + u0(i,j, k-1)+u0(i,j, k+1) - h*h*f(x(i),y(j), z(k));
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