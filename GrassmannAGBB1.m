function [X, out] = GrassmannAGBB1(fun, X0, opts, varargin)
% Exponential-based accelerated gradient method with BB step
% on Grassmann manifold
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
w = opts.w;

% initialization
[~, p] = size(X0);
alpha = 1/L;
Z = X0;
Y = Z;
fy = -inf;

% record of norm of gradient
recg = [];

% restart trigger
trig = 1;

% main loop
for k = 1:mxitr
    
    lambda = 2/(k+1);
    
    % compute Eta and X
    if trig
        X = Z;
    else
        Eta = (1 - lambda)*LogGr(Z, Y, 1);
        [X, U, V, c, s] = ExpGr(Z, Eta);
    end
    
    % compute function value and gradient at X
    [fx, Gx] = feval(fun, X, varargin{:});
    Gradx = ProjGr(X, Gx);
    nrmg = norm(Gradx,'fro');
    if k == 1
        fz = fx;
    end
    
    % update record
    recg_new = [recg; nrmg];
    recg = recg_new;
    
    % convergence test
    if nrmg < gtol
        % iteration
        iter = k - 1;
        break
    end
    
    % inverse vector transport of gradient at X
    if trig
        Xi = Gradx;
    else
        Xi = invParaGr(Gradx, U, V, c, s, Z);
    end
    
    %beta = w*alpha;
    beta = min(100,max(1.5,sqrt(k)*w))*alpha;
    
    % compute Y and Z with backtracking line search
    count = 0;
    while count < 5
        Y = ExpGr(X, -alpha*Gradx);
        [fyt, Gy] = feval(fun, Y, varargin{:});
        ratio = (max(fx,fy) - fyt)/(alpha*nrmg^2);
        if ratio >= nu
            break
        else
            count = count + 1;
            alpha = alpha/mu;
            beta = beta/mu;
        end
    end
    fy = fyt;
    
    Z = ExpGr(Z, -beta*Xi);
    [fzt, ~] = feval(fun, Z, varargin{:});
    if fzt > max([fx,fy,fz])
        trig = 1;
        Z = Y;
        %w = max(1.01, 3/4*w);
        w = max(0.1, 3/4*w);
        fz = fy;
    else
        trig = 0;
        %w = min(100, 4/3*w);
        w = min(1, 4/3*w);
        fz = fzt;
    end
    
    % BB stepsize
    S = alpha*Gradx;
    Grady = ProjGr(Y, Gy); 
    W = Gradx - Grady; 
    SW = abs(sum(sum(S.*W)));
    if mod(k,2)==0
        alpha = sum(sum(S.*S))/SW;
    else
        alpha = SW/sum(sum(W.*W)); 
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
out.fval = fx;
out.nrmg = nrmg;
out.recg = recg;
out.feasi = feasi;