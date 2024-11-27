function [f, G] = funhetero(X, A)
% Sum of heterogeneous quadratic functions
[n, p] = size(X);
G = zeros(n, p);
f = 0;
for i = 1:p
    Ai = A(:, n*(i-1)+1:n*i);
    Xi = X(:, i);
    if isdiag(Ai)
        Gi = bsxfun(@times, diag(Ai), Xi);
    else
         Gi = Ai*Xi;
    end
    G(:,i) = Gi;
    f = f + Gi'*Xi;
end
f = 0.5*f;