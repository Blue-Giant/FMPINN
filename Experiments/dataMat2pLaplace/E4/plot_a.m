function plot_a(geom, a)

ftsz = 14;
       
X = geom.pt(1,:); Y = geom.pt(2,:);
nn = sqrt(length(X));
X = reshape(X, nn, nn); Y = reshape(Y, nn, nn);
a = reshape(a, nn, nn);
% surf(X, Y, log10(a), 'edgecolor', 'none');
surf(X, Y, log10(a));
colorbar;
set(gca, 'XMinortick', 'off', 'YMinorTick', 'off', 'Fontsize', ftsz);
set(gcf, 'Renderer', 'zbuffer');
end