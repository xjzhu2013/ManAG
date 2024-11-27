function Xi = invTranSt(Zeta, U, V, W)
% Cayley transform inverse vector transport on Stiefel manifold
p = size(Zeta, 2);
Xi = Zeta - U*((eye(2*p) + 0.5*W)\(V'*Zeta));