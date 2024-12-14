function [X, out] = GrassmannAG2(fun, X0, opts, varargin)
% Retraction-based accelerated gradient method on Grassmann manifold
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
w = opts.w; 

% initialization
[~, p] = size(X0);
alpha = 1/L;
Z = X0;
Y = Z;

% record of norm of gradient
recg = [];

% restart trigger
trig = 1;

% main loop
for k = 1:mxitr
    
    lambda = 2/(k+1); 
    
    % previous X and gradient
    if k > 1
        Xp = X;
        Gradp = Grad;
    end
    
    % compute Eta and X
    if trig
        X = Z;
    else
        Eta = (1 - lambda)*invRetrGr(Z, Y);
        X = RetrGr(Z, Eta);
    end
    
    % compute function value and gradient at X
    [f, G] = feval(fun, X, varargin{:});
    Grad = ProjGr(X, G);
    nrmg = norm(Grad,'fro');
    
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
        Xi = Grad;
    else
        Xi = invTranGr(Grad, Eta, Z);
    end
    
    % BB stepsize
    if k > 1
        %S = invRetrSt(X, Xp);
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
    end
    
    beta = (1+w*lambda)*alpha;
    
    % compute Y 
    Y = RetrGr(X, -alpha*Grad);
    
    % compute Z with restart
    if mod(k,10)==0
        trig = 1;
        Z = Y;
    else
        trig = 0;
        Z = RetrGr(Z, -beta*Xi);
    end

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