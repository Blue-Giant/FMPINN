clc;
clear all
close all
meshData = load('meshXY6.mat');
meshXY = meshData.meshXY;
xt = meshXY(1,1:end);
yt = meshXY(2,1:end);

q = 6;
nl = 2;
T = [];
J = [];

% geom: (only square geometry available now)
% generating 2d square mesh for the region [-1, 1] x [-1 1]
geom.q = q;
geom.nl = nl;
geom.L = 2; % side length 
geom.dim = 2; % dimension of the problem
geom.m = 2^geom.dim; % 
geom.N1 = 2^q; % dofs in one dimension
geom.N = (geom.m)^geom.q; % dofs in the domain
geom.h = geom.L/(geom.N1+1); % grid size
geom.xstart = -1;
geom.xend = 1;
geom.ystart = -1;
geom.yend = 1;

geom = assemble_fmesh(geom);

% Aeps = ones(size(xt));
% for k = 1:q
%     Aeps = Aeps.*(1+0.5*cos(2^k*pi*(xt+yt))).*...
%         (1+0.5*sin(2^k*pi*(yt-3*xt)));
% end

data2Aeps = load('Aeps.mat');
Aeps = data2Aeps.aeps;

figure('name','Aeps')
plot_a(geom,Aeps);
hold on



