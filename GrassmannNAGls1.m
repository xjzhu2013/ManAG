function [X, out] = GrassmannNAGls1(fun, X0, opts, varargin)
% Exponential-based Nesterov accelerated gradient method on Grassmann 
% manifold with line search
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
Gradx = ProjGr(X, Gx);
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
        Yt = ExpGr(X, -alpha*Gradx);
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
    Xi = LogGr(Yt, Y, 1);
    
    % update Y
    Y = Yt;
    
    % compute X with its function value and gradient
    X = ExpGr(Y, ((1-t)/t_new)*Xi);
    [fxt, Gx] = feval(fun, X, varargin{:});
    
    % restart or not
    if fxt > max(fx,fy)
        X = Y;
        fx = fy;
        trig = 1;
    else
        fx = fxt;
        trig = 0;
    end
    
    % update t
    t = t_new;
    
    % BB stepsize
    S = -alpha*Gradx;
    %S = Y - X;
    Grady = ProjGr(Y, Gy); 
    H = ProjSt(X,Grady) - Gradx; 
    %H = Grady - Gradx; 
    SH = abs(sum(sum(S.*H)));
    if mod(k,2)==0
        alpha = sum(sum(S.*S))/SH;
    else
        alpha = SH/sum(sum(H.*H)); 
    end
    alpha = max(min(alpha, 1e20), 1e-20);
    
    % update gradient and its norm
    if trig
        Gradx = Grady;
    else
        Gradx = ProjGr(X, Gx);
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