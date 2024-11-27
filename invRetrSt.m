function Eta = invRetrSt(X, Y)
% Cayley transform inverse retraction on Stiefel manifold
p = size(X, 2);
M = eye(p) + X'*Y;
Eta = 2*(Y/M + X/(M') - X);