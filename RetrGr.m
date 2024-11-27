function [Y, M, U] = RetrGr(X, Eta)
% Cayley transform retraction on Grassmann manifold
p = size(X, 2);
EtE = Eta'*Eta;
M = eye(p) + 0.25*EtE;
U = X + 0.5*Eta;
Y = X + Eta - 0.5*U*(M\EtE);
    