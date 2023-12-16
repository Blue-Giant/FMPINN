clc;
clear all
close all

q = 7;
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
geom.xstart = 0;
geom.xend = 1;
geom.ystart = 0;
geom.yend = 1;

geom = assemble_fmesh(geom);


solu_data2U = load('test_solus2U');
Solu_UTrue = solu_data2U.Utrue;
Solu_UDNN = solu_data2U.UNN;
figure('name','Exact_Solu')
mesh_solu2u = plot_fun2in(geom,Solu_UTrue);
hold on

figure('name','DNN_Solu')
mesh_solu_unn = plot_fun2in(geom,Solu_UDNN);
hold on

err2solu = abs(Solu_UTrue-Solu_UDNN);
figure('name','Err2solu')
mesh_solu_err2u = plot_fun2in(geom,err2solu);
title('Absolute Error for U')
hold on
colorbar;
caxis([0 0.0025])
hold on


solu_data2V = load('test_solus2V');
Solu_VTrue = solu_data2V.Vtrue;
Solu_VDNN = solu_data2V.VNN;
figure('name','Exact_V')
mesh_solu2v = plot_fun2in(geom,Solu_VTrue);
hold on

figure('name','VNN_Solu')
mesh_solu_vnn = plot_fun2in(geom,Solu_VDNN);
hold on

err2solu_v = abs(Solu_VTrue-Solu_VDNN);
figure('name','Err2V')
mesh_solu_err2v = plot_fun2in(geom,err2solu_v);
title('Absolute Error for V')
hold on
colorbar;
caxis([0 0.0025])
hold on


solu_data2P = load('test_solus2P');
Solu_PTrue = solu_data2P.Ptrue;
Solu_PDNN = solu_data2P.PNN;
figure('name','Exact_P')
mesh_solu2Ptrue = plot_fun2in(geom,Solu_PTrue);
hold on

figure('name','PNN_Solu')
mesh_solu2PNN = plot_fun2in(geom,Solu_PDNN);
hold on

err2solu_P = abs(Solu_PTrue-Solu_PDNN);
figure('name','Err2P')
mesh_err2P = plot_fun2in(geom,err2solu_P);
title('Absolute Error for P')
hold on
colorbar;
caxis([0 0.1])
hold on
