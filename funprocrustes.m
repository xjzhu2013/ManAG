function [f, G] = funprocrustes(X, A, B)
R = A*X - B;
f = 0.5*norm(R,'fro')^2;
G = A'*R;