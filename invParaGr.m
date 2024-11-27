function Xi = invParaGr(Zeta, U, V, c, s, X)
% inverse parallel translation on Grassmann manifold
XtZeta = X'*Zeta;
Xi = Zeta - X*(XtZeta) + U*(bsxfun(@times,(1-c).*c,(U'*Zeta))...
     - bsxfun(@times,(1-c).*s,V'*XtZeta));