function [f, G] = funquad(X, A, B)
% Quadratic function
AX = A*X;
G = AX + B;
f = sum(dot(0.5*AX + B, X, 1));