function [Y, U, V, c, s] = ExpGr(X, Eta)
% exponential on Grassmann manifold
[U, S, V] = svd(Eta, 0);
d = diag(S);
c = cos(d);
s = sin(d);
Y = (bsxfun(@times, X*V, c') + bsxfun(@times, U, s'))*V';