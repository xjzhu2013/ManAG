function [X, out] = GrassmannAGls2(fun, X0, opts, varargin)
% Retraction-based accelerated gradient method on Grassmann manifold with
% line search
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
    beta = (1+w*lambda)*alpha;
    
    % compute Eta and X
    if trig
        X = Z;
    else
        Eta = (1 - lambda)*invRetrGr(Z, Y);
        X = RetrGr(Z, Eta);
    end
    
    % compute function value and gradient at X
    [fx, Gx] = feval(fun, X, varargin{:});
    Gradx = ProjGr(X, Gx);
    nrmg = norm(Gradx,'fro');
    
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
        Xi = invTranGr(Gradx, Eta, Z);
    end
    
    % compute Y with backtracking line search
    count = 0;
    while count < 5
        Y = RetrGr(X, -alpha*Gradx);
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
    
    % compute Z
    Z = RetrGr(Z, -beta*Xi);
    
    % restart or not
    if mod(k,10)==0
        trig = 1;
        Z = Y;
    else
        trig = 0;
    end

    % BB stepsize
    S = -alpha*Gradx;
    %S = Y - X;
    Grady = ProjGr(Y, Gy); 
    H = ProjGr(X,Grady) - Gradx; 
    %H = Grady - Gradx; 
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
out.fval = fx;
out.nrmg = nrmg;
out.recg = recg;
out.feasi = feasi;