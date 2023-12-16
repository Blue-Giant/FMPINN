clear all
close all
clc
data2mesh = load('meshXY6.mat');
meshxy = data2mesh.meshXY;
xt = meshxy(1,1:end);
yt = meshxy(2,1:end);
q = 6;

% Aeps1=(1+0.5*cos(2*pi*(meshx+meshy))).*(1+0.5*sin(2*pi*(meshy-3*meshx)));
% Aeps2=(1+0.5*cos(4*pi*(meshx+meshy))).*(1+0.5*sin(4*pi*(meshy-3*meshx)));
% Aeps3=(1+0.5*cos(8*pi*(meshx+meshy))).*(1+0.5*sin(8*pi*(meshy-3*meshx)));
% Aeps4=(1+0.5*cos(16*pi*(meshx+meshy))).*(1+0.5*sin(16*pi*(meshy-3*meshx)));
% Aeps5=(1+0.5*cos(32*pi*(meshx+meshy))).*(1+0.5*sin(32*pi*(meshy-3*meshx)));
% Aeps6=(1+0.5*cos(64*pi*(meshx+meshy))).*(1+0.5*sin(64*pi*(meshy-3*meshx)));
% Aeps = Aeps1.*Aeps2.*Aeps3.*Aeps4.*Aeps5.*Aeps6;

Aeps = ones(size(xt));
for k = 1:q
    Aeps = Aeps.*(1+0.5*cos(2^k*pi*(xt+yt))).*...
        (1+0.5*sin(2^k*pi*(yt-3*xt)));
end

figure('name', 'Aeps')
plot3(xt,yt,Aeps,'b.')
hold on
