function Eta = invRetrGr(X, Y)
% Cayley transform inverse retraction on Grassmann manifold
[U, S, V] = svd(X'*Y);
s = diag(S);
Eta = 2*(Y*V - X*bsxfun(@times, U, s'))*bsxfun(@rdivide, U', s + 1);