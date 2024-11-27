function Eta = LogGr(X, Y, method)
% Logarithm on Grassmann manifold
switch method
    case 1
        [U, ~, V] = svd(Y'*X);
        Y = Y*(U*V');
        [U, S, V] = svd(Y - X*(X'*Y), 0);
        s = asin(diag(S));
        Eta = U*bsxfun(@times, V', s);
    case 2
        XtY = X'*Y;
        M = (Y - X*XtY)/XtY;
        [U, S, V] = svd(M, 0);
        s = diag(S);
        Eta = U*bsxfun(@times, V', atan(s));
end
