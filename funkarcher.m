function [f, G] = funkarcher(X, Y)
% Cost function of weighted Karcher mean of subspaces
[n, p] = size(X);
m = size(Y, 2)/p;
w = ones(1,m)/m;
f = 0;
G = zeros(n, p);
for i = 1:m
    Yi = Y(:,(i-1)*p+1:i*p);
    Di = LogGr(X, Yi, 1);
    f = f + 0.5*w(i)*norm(Di, 'fro')^2;
    G = G - w(i)*Di;
end