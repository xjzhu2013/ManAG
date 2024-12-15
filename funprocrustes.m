function [f, G] = funprocrustes(X, A, B)
% Procrustes cost function
R = A*X - B;
f = 0.5*norm(R,'fro')^2;
G = A'*R;