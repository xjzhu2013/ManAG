function [X, out] = GrassmannNAG1(fun, X0, opts, varargin)
% Exponential-based Nesterov accelerated gradient method on Grassmann manifold
% Input: fun - objective 
%        X0 - initial point
%        opts - options for parameters
%        varargin - data in objective
% Output: X - solution
%         out - outcome information

% parameters
L = opts.L;
gtol = opts.gtol;
mxitr = opts.mxitr;

% initialization
[~, p] = size(X0);
alpha = 1/L;
X = X0;
Y = X0;
t = 1;

% compute function value and gradient at X
[f, G] = feval(fun, X, varargin{:});
Grad = ProjGr(X, G);
nrmg = norm(Grad,'fro');

% record of norm of gradient
recg = nrmg;

% main loop
for k = 1:mxitr
    
    % convergence test
    if nrmg < gtol
        % iteration
        iter = k - 1;
        break
    end
    
    % compute Y 
    Yt = ExpGr(X, -alpha*Grad);
    
    % compute new accelerate parameter  
    t_new = 0.5*(1 + sqrt(1+4*t^2));
    
    % compute inverse retraction
    Xi = LogGr(Yt, Y, 1);
    
    % update Y
    Y = Yt;
    
    % previous X and gradient
    Xp = X;
    Gradp = Grad;
    
    % compute X with restart
    if mod(k,10)==0
        X = Y;
    else
        X = ExpGr(Y, ((1-t)/t_new)*Xi);
    end
    
    % update t
    t = t_new;
    
    % compute function value and gradient at X
    [f, G] = feval(fun, X, varargin{:});
    Grad = ProjGr(X, G);
    nrmg = norm(Grad,'fro');
    
    % update record of norm of gradient 
    recg_new = [recg; nrmg];
    recg = recg_new;
    
    % BB stepsize
    %S = LogGr(X, Xp, 1);
    S = Xp - X;
    H = ProjGr(X,Gradp) - Grad;
    %H = Gradp - Grad; 
    SH = abs(sum(sum(S.*H)));
    if mod(k,2)==0
        alpha = sum(sum(S.*S))/SH;
    else
        alpha = SH/sum(sum(H.*H));
    end
    alpha = max(min(alpha, 1e20), 1e-20);
    
    % iteration
    iter = k;
    
end

% re-orthogonalization
XtX = X'*X; 
feasi = norm(XtX - eye(p), 'fro'); 
if feasi > 1e-13
    X = X*(XtX)^(-1/2);
    %[U, ~, V] = svd(X, 0); X = U*V'; 
    feasi = norm(X'*X - eye(p), 'fro'); 
end

% output
out.iter = iter;
out.fval = f;
out.nrmg = nrmg;
out.recg = recg;
out.feasi = feasi;