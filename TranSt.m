function Zeta = TranSt(Xi, U, V, W)
% Cayley transform vector transport on Stiefel manifold
p = size(Xi, 2);
Zeta = Xi + U*((eye(2*p) - 0.5*W)\(V'*Xi)); 