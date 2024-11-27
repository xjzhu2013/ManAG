function Zeta = TranGr(Eta, Xi, M, U)
% Cayley transform vector transport on Grassmann manifold
Zeta = Xi - U*(M\(Eta'*Xi));