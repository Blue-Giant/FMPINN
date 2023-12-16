load K1 % SPE10 coefficients 60 x 220 x 85
K = K1(1:60, 1:60, 1);
xt = pt(1,:); yt = pt(2,:);
xmin = -1; xmax = 1; ymin = -1; ymax = 1;
dx = (xmax-xmin)/60;
dy = (ymax-ymin)/60;
a=zeros(size(xt));     % added by Zhu, preallocate for 
for i = 1:length(xt)
    j = floor((xt(i)-xmin)/dx) + 1;
    k = floor((yt(i)-ymin)/dy) + 1;
    a(i) = 10^K(j,k);
end