function [Y, U, V, W] = RetrSt(X, Eta)
% Cayley transform retraction on Stiefel manifold
p = size(X, 2);
D = Eta - 0.5*X*(X'*Eta);
U = [D, X];
V = [X, -D];
W = V'*U;
VtX = W(:,p+1:end);
Y = X + U*((eye(2*p) - 0.5*W)\VtX); 