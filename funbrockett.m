function [f, G] = funbrockett(X, A, d)
% Brockett cost function
G = A*bsxfun(@times, X, d);
f = 0.5*sum(dot(X, G, 1));