% script file for illustration of FEM for 1d oscillatory diffusion
clear all
close all
clc

% define Interval
I = [0,1];

% define diffusivity
% epsilon = 2^(-15);
% epsilon = 0.1;
% epsilon = 0.05;
% epsilon = 1.0/4;
% epsilon = 1.0/8;
% epsilon = 1.0/16;
% epsilon = 1.0/32;
% epsilon = 1.0/64;
epsilon = 1.0/128;
% A = @(x) (0.5+x.^2)./(2+sin(2*pi*x./epsilon));
A = @(x) (1+x.^2)./(2+sin(2*pi*x./epsilon));

M = 10000;

% plot coefficient
figure(1)
t_step=1.0/M;
t = 0:t_step:1;
kkk = [0.5 0.1 0.6];
hcoeff = plot(t,A(t), 'color', kkk, 'LineWidth', 2)
% hcoeff = plot(t,A(t), 'b', 'LineWidth', 2)
hold on
set(gca, 'Fontsize', 18)
axis([0 1 0 1.2*max(A(t))]);
xlabel('$x$', 'Fontsize', 18, 'Interpreter', 'latex')
ylabel('$A^{\varepsilon}(x)$', 'Fontsize', 18, 'Interpreter', 'latex')
%latexprint(strcat('../../LectureNotes/gfx/oscdiff1d_coeff_',num2str(-log2(epsilon))), '-r600')

% forcing term
f = @(x) 5*cos(pi*x);
% f = @(x) ones(size(x));
%f = @(x) x.*(1-x);

% finite element approximation on equi-distant grid
for k=1:15
    % finte element grid
    hx = 1/(M-1); % mesh size
    x = (0:hx:1)';

    % stiffness matrix
    for j=1:M-1
        Amean(j)=quadgk(@(x) A(x),x(j),x(j+1), 'RelTol', 1e-8)./hx;
    end
    
    Ndof = M-2;
    S = sparse(1:Ndof,1:Ndof,1/hx.*(Amean(1:M-2)+Amean(2:M-1)),Ndof,Ndof);
    S = S+sparse(2:Ndof,1:Ndof-1,-1/hx.*Amean(2:M-2),Ndof,Ndof);
    S = S+sparse(1:Ndof-1,2:Ndof,-1/hx.*Amean(2:M-2),Ndof,Ndof);
    
    % right hand side
    rhs = hx*f(x(2:M-1));
    
    % solve
    u = S\rhs;
    u = [0;u;0]; % extend to boundary
    
    % plot approximation on current mesh
%     figure(2)
%     hfem(k) = plot(x,uk, 'b', 'LineWidth', 1);
    %latexprint(strcat('../../LectureNotes/gfx/oscdiff1d_fem_',...num2str(k), '_', num2str(-log2(epsilon))), '-r600')
end

figure(2)
hfem = plot(x,u, 'b', 'LineWidth', 1.25);
hold on
set(gca, 'Fontsize', 18)
axis([0 1 min(u)-0.05 1.2*max(u)]);
xlabel('$x$', 'Fontsize', 18, 'Interpreter', 'latex')
ylabel('$u^{\varepsilon}(x)$', 'Fontsize', 18, 'Interpreter', 'latex')
hold on