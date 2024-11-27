function [f, G] = funeigen(X, A)
% Linear eigenvalue cost function
G = A*X;
f = 0.5*sum(dot(X, G, 1));