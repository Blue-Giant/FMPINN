function data = calA(geom, data, opts)
%CALA Calculate the elliptic(椭圆形的；省略的) coefficient a(x)
%   a(x) defined on the center of the square element, whose coordinates are
%   stored in pc

pt = geom.pt;

switch opts.eqn.a
    case 1 % Laplacian coefficient
        a = ones(size(pt, 2), 1);
    case 2 % trignometric coefficient
        xt = pt(1,:); yt = pt(2,:);
        a = 2+sin(3*pi*xt).*cos(5*pi*yt);
    case 3 % multiscale trignometric coefficient
        xt = pt(1,:); yt = pt(2,:);
        e1=1/5; e2=1/13; e3=1/17; e4=1/31; e5=1/65;
        a=1/6*((1.1+sin(2*pi*xt/e1))./(1.1+sin(2*pi*yt/e1))...
            +(1.1+sin(2*pi*yt/e2))./(1.1+cos(2*pi*xt/e2))...
            +(1.1+cos(2*pi*xt/e3))./(1.1+sin(2*pi*yt/e3))...
            +(1.1+sin(2*pi*yt/e4))./(1.1+cos(2*pi*xt/e4))...
            +(1.1+cos(2*pi*xt/e5))./(1.1+sin(2*pi*yt/e5))+sin(4*xt.^2.*yt.^2)+1);
    case 4 % Gamblet paper coefficient
        xt = pt(1,:); yt = pt(2,:);
        a = ones(size(xt));
        for k = 1:geom.q
            geom.q;
            a = a.*(1+0.5*cos(2^k*pi*(xt+yt))).*...
                (1+0.5*sin(2^k*pi*(yt-3*xt)));
        end
    case 5 % periodic checker board
        m = 16;
        A = 10;
        xt = pt(1,:); yt = pt(2,:);
        xmin = -1; xmax = 1; ymin = -1; ymax = 1;
        dx = (xmax - xmin)/m;
        dy = (ymax - ymin)/m;
        a=zeros(size(xt)); % added by Zhu, preallocate for 
        for i = 1:length(xt)
            j = floor((xt(i)-xmin)/dx) + 1;
            k = floor((yt(i)-ymin)/dy) + 1;
            a(i) = mod(j+k,2)*(A-1/A) + 1/A;
        end
    case 6 % random multiscale 
        seed='default';  
        xt = pt(1,:); yt = pt(2,:);
        n = length(xt);
        m=floor(log2(sqrt(n)))+1;
        r=2;
        se=rng(seed);           % add by Zhu, gurantee repeatable results
        for k = 1:m
            A{k}=1/(1+r)+(1+r-1/(1+r))*rand(2^k);
        end

        % establish a cartesian grid
        xmin = -1; xmax = 1; ymin = -1; ymax = 1;
        dx=(xmax-xmin)/2^m;
        dy=(ymax-ymin)/2^m;
        
        c=ones(2^m);
        for i=1:m
            X=A{i};
            Y=ones(2^(m-i));
            c=c.*kron(X,Y);
        end
        a=zeros(size(xt));     %modified by Zhu to    
        for k=1:n
            i=floor((xt(k)-xmin)/dx)+1;
            j=floor((yt(k)-ymin)/dy)+1;
            a(k)=c(i,j);
        end
        a=a+1; % modified by Zhu to remove the case contrast is infinity
    case 7 % random checker board
        m = 64; A = sqrt(400);
        xt = pt(1,:); yt = pt(2,:);
        xmin = -1; xmax = 1; ymin = -1; ymax = 1;
        dx = (xmax - xmin)/m;
        dy = (ymax - ymin)/m;
        a=zeros(size(xt)); % added by Zhu, preallocate for
        rng('default')
        r = binornd(1, 0.5, m^2, 1);
        for i = 1:length(xt)
            j = floor((xt(i)-xmin)/dx) + 1;
            k = floor((yt(i)-ymin)/dy) + 1;
            a(i) = r((j-1)*m+k)*(A-1/A) + 1/A;
        end
    case 8 % SPE10
        load K1 % SPE10 coefficients 60 x 220 x 85
        K = K1(1:60, 1:60, 1);
        xt = pt(1,:); yt = pt(2,:);
        xmin = -1; xmax = 1; ymin = -1; ymax = 1;
        dx = (xmax-xmin)/60;
        dy = (ymax-ymin)/60;
        a=zeros(size(xt)); % added by Zhu, preallocate for 
        for i = 1:length(xt)
            j = floor((xt(i)-xmin)/dx) + 1;
            k = floor((yt(i)-ymin)/dy) + 1;
            a(i) = 10^K(j,k);
        end
end

data.a = a;

end

