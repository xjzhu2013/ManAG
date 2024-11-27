function P = ProjSt(X, Z)
% Projection onto tangent space to Stiefel manifold
XtZ = X'*Z;
P = Z - 0.5*X*(XtZ + XtZ');