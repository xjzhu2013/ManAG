function [X, out] = GrassmannGrad(fun, X0, opts, varargin)
% Basic gradient method on Grassmann manifold
% Input: fun - objective 
%        X0 - initial point
%        opts - options for parameters
%        varargin - data in objective
% Output: X - solution
%         out - outcome information

% parameters
L = opts.L;
mu = opts.mu;
nu = opts.nu;
gtol = opts.gtol;
mxitr = opts.mxitr;
method = opts.method;

% initialization
[~, p] = size(X0);
alpha = 1/L;
X = X0;

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

    % backtracking line search
    count = 0;
    while count < 5
        switch method
            case 1
                % exponential
                Xt = ExpGr(X, -alpha*Grad);
            case 2
                % retraction
                Xt = RetrGr(X, -alpha*Grad);
        end
        [ft, Gt] = feval(fun, Xt, varargin{:});
        ratio = (f - ft)/(alpha*nrmg^2);
        if ratio >= nu
            break
        else
            count = count + 1;
            alpha = alpha/mu;
        end
    end

    alpha = 1/L;
    
    % update X, function value and gradient at X
    X = Xt;
    f = ft;
    Grad = ProjGr(X, Gt);
    nrmg = norm(Grad,'fro');
    
    % update record
    recg_new = [recg; nrmg];
    recg = recg_new;
    
    % iteration
    iter = k;
    
end

%re-orthogonalization
XtX = X'*X;
feasi = norm(XtX - eye(p), 'fro'); 
if feasi > 1e-13
    X = X*(XtX)^(-1/2);
    %[U, ~, V] = svd(Xt, 0); Xt = U*V'; 
    feasi = norm(X'*X - eye(p), 'fro'); 
end

% output
out.iter = iter;
out.fval = f;
out.nrmg = nrmg;
out.recg = recg;
out.feasi = feasi;