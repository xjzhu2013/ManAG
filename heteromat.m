function A = heteromat(n, p)
m = n/2;
A = [];
for i = 1:p
    B = gallery('tridiag',-ones(m-1,1), 2*ones(m,1), -ones(m-1,1));
    E = 1e-6*randn(n);
    Ai = [B, zeros(m); zeros(m), zeros(m)] + (E + E');
    A1 = [A, Ai];
    A = A1;
end