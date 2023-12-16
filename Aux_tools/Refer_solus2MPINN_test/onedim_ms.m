% script file for illustration of FEM for 1d oscillatory diffusion
clear all
close all

% define Interval
I = [0,1];

% define diffusivity
% epsilon = 2^(-15);
epsilon = 0.001;
A = @(x) 1./(2+cos(2*pi*x./epsilon));

% plot coefficient
figure(1)
t = 0:0.0001:1;
hcoeff = plot(t,A(t), 'b', 'LineWidth', 2)
hold on
set(gca, 'Fontsize', 18)
axis([0 1 0 1.2*max(A(t))]);
xlabel('$x$', 'Fontsize', 18, 'Interpreter', 'latex')
ylabel('$A_{\varepsilon}(x)$', 'Fontsize', 18, 'Interpreter', 'latex')
%latexprint(strcat('../../LectureNotes/gfx/oscdiff1d_coeff_',num2str(-log2(epsilon))), '-r600')

% forcing term
f = @(x) ones(size(x));

% solution
uepsilon = @(x) x - x.^2 ...
    + epsilon*(-1/(2*pi)*x.*sin(2*pi*x/epsilon)...
    + 1/(4*pi)*sin(2*pi*x/epsilon)...
    - epsilon/(4*pi^2)*cos(2*pi*x/epsilon)+...
    + epsilon/(4*pi^2));

% plot solution
figure(2)
hsol = plot(t,uepsilon(t), 'r', 'LineWidth', 2);
hold on
set(gca, 'Fontsize', 18)
axis([0 1 0 1.2*max(uepsilon(t))]);
xlabel('$x$', 'Fontsize', 18, 'Interpreter', 'latex')
ylabel('$u_\epsilon(x)$', 'Fontsize', 18, 'Interpreter', 'latex')
%latexprint(strcat('../../LectureNotes/gfx/oscdiff1d_sol_',num2str(-log2(epsilon))), '-r600')

% finite element approximation on equi-distant grid
for k=1:15
    % finte element grid
    N = 1+2^k;      % number of grid points
    h(k) = 1/(N-1); % mesh size
    x = (0:h(k):1)';

    % stiffness matrix
    for j=1:N-1
        Amean(j)=quadgk(@(x) A(x),x(j),x(j+1), 'RelTol', 1e-8)./h(k);
    end
    
    Ndof = N-2;
    S = sparse(1:Ndof,1:Ndof,1/h(k).*(Amean(1:N-2)+Amean(2:N-1)),Ndof,Ndof);
    S = S+sparse(2:Ndof,1:Ndof-1,-1/h(k).*Amean(2:N-2),Ndof,Ndof);
    S = S+sparse(1:Ndof-1,2:Ndof,-1/h(k).*Amean(2:N-2),Ndof,Ndof);
    
    % right hand side
    rhs = h(k)*f(x(2:N-1));
    
    % solve
    uk = S\rhs;
    uk = [0;uk;0]; % extend to boundary
    
    % plot approximation on current mesh
%     figure(2)
%     hfem(k) = plot(x,uk, 'b', 'LineWidth', 1);
    %latexprint(strcat('../../LectureNotes/gfx/oscdiff1d_fem_',...num2str(k), '_', num2str(-log2(epsilon))), '-r600')
end

figure(2)
hfem(k) = plot(x,uk, 'b', 'LineWidth', 1);
hold on

% estimate error
e = uepsilon(x)-uk;
err(k) = 0;
for j=1:N-1
    err(k) = err(k) + quadgk(@(s) (uepsilon(s)-uk(j)...
        - (uk(j+1)-uk(j))./h(k)*(s-x(j))).^2,...
        x(j),x(j+1), 'RelTol', 1e-8);
end
err(k) = sqrt(err(k));

% plot error
figure(3)
herr = loglog(h,err, '-bd', 'LineWidth',2)
hold on
set(gca, 'Fontsize', 18, 'xscale', 'log', 'yscale', 'log',...
    'xtick',10.^(-4:1:0), 'ytick', 10.^(-8:2:0))