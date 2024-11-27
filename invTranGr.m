function Xi = invTranGr(Zeta, Eta, X)
% Cayley transform inverse vector transport on Grassmann manifold
Xi = Zeta - (X + 0.5*Eta)*(X'*Zeta);