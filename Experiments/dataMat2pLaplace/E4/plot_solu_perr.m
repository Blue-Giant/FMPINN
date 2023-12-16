clc;
clear all
close all
meshData = load('meshXY6.mat');
meshXY = meshData.meshXY;

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


figure('name','true')
data2utrue = load('u_true6.mat');
utrue = data2utrue.u_true;
mesh_true = plot_fun(geom,utrue);
hold on



