clear all
close all
clc

N = 64;
xl = 0; xr = 1;
yl = 0; yr = 1;
zl = 0; zr = 1;
h = (xr-xl)/N;

x = xl:h:xr;
y = yl:h:yr;
z = zl:h:zr;

data = load('uref64.mat');
u = data.u;

u_slice = reshape(u(21,:,:),N+1,N+1);
figure(1)
surf(y,z,u_slice);
% title('Approx');

eps = 0.1;
Aeps = @(XX,YY,ZZ)  2.0+sin(pi*XX/eps)*sin(pi*YY/eps)*sin(pi*ZZ/eps);

A = zeros(N+1, N+1, N+1);
for i=1:N+1
    for j=1:N+1
        for k=1:N+1
        A(i,j,k) = Aeps(x(i),y(j),z(k));
        end
    end
end

A_slice = reshape(A(21,:,:),N+1,N+1);
figure('name','Aeps')
surf(y,z,A_slice);
% title('Aeps');