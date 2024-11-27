function Zeta = ParaGr(X, U, V, c, s, Xi)
% parallel translation on Grassmann manifold
Zeta = Xi - (X*bsxfun(@times, V, s') + bsxfun(@times, U, 1 - c'))*(U'*Xi);