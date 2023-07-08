clear all
close all
clc
data2mesh = load('meshXY6.mat');
meshxy = data2mesh.meshXY;
meshx = meshxy(1,1:end);
meshy = meshxy(2,1:end);

utrue=(sin(pi*meshx).*sin(pi*meshy)+0.05*sin(20*pi*meshx).*sin(20*pi*meshy));
figure('name', 'ueps')
plot3(meshx,meshy,utrue,'b.')
hold on

% zcos=0.25*(2+0.5*cos(5*pi*meshx).*cos(10*pi*meshy)+0.5*cos(15*pi*meshx).*cos(20*pi*meshy));
% figure('name', 'Aeps')
% surface(meshx,meshy,zcos)
% hold on
% figure('name', 'uAeps')
% zsin_cos = zsin.*zcos;
% surface(meshx,meshy,zsin_cos)