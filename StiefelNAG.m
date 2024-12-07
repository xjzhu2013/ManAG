function [X, out] = StiefelNAG(fun, X0, opts, varargin)
% Retraction-based Nesterov accelerated gradient method on Stiefel manifold
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

% initialization
[~, p] = size(X0);
alpha = 1/L;
X = X0;
Y = X0;
fy = -inf;
t = 1;

% compute function value and gradient at X
[fx, Gx] = feval(fun, X, varargin{:});
Gradx = RGradSt(X, Gx);
nrmg = norm(Gradx,'fro');

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
    
    % compute Y with backtracking line search
    count = 0;
    while count < 5
        Yt = RetrSt(X, -alpha*Gradx);
        [fyt, Gy] = feval(fun, Yt, varargin{:});
        ratio = (max(fx,fy) - fyt)/(alpha*nrmg^2);
        if ratio >= nu
            break
        else
            count = count + 1;
            alpha = alpha/mu;
        end
    end
    fy = fyt;
    
    % compute new accelerate parameter  
    t_new = 0.5*(1 + sqrt(1+4*t^2));
    
    % compute inverse retraction
    Xi = invRetrSt(Yt, Y);
    
    % update Y
    Y = Yt;
    
    % compute X with its function value and gradient
    X = RetrSt(Y, ((1-t)/t_new)*Xi);
    [fxt, Gx] = feval(fun, X, varargin{:});
    if fxt > max(fx,fy)
        X = Y;
        fx = fy;
        % restart
        trig = 1;
    else
        fx = fxt;
        % no restart
        trig = 0;
    end
    
    % update t
    t = t_new;
    
    % BB stepsize
    S = alpha*Gradx;
    Grady = RGradSt(Y, Gy); 
    W = Gradx - Grady; 
    SW = abs(sum(sum(S.*W)));
    if mod(k,2)==0
        alpha = sum(sum(S.*S))/SW;
    else
        alpha = SW/sum(sum(W.*W)); 
    end
    alpha = max(min(alpha, 1e20), 1e-20);
    
    % update gradient and its norm
    if trig
        Gradx = Grady;
    else
        Gradx = RGradSt(X, Gx);
    end
    nrmg = norm(Gradx,'fro');
    
    % update record of norm of gradient 
    recg_new = [recg; nrmg];
    recg = recg_new;
    
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
out.fval = fx;
out.nrmg = nrmg;
out.recg = recg;
out.feasi = feasi;