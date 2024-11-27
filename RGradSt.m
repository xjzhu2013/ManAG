function Grad = RGradSt(X, G)
% Transform Euclidean gradient to Riemannian gradient under the canonical 
% metric on Stiefel manifold
Grad = G - X*(G'*X);