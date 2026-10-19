#On va se placer dans un contexte où (Y,X,W,B) sont des matrices et on a une relation du type Y=XW + B
# Si Y est mxn => X est mxq W est qxn et B est mxn
import torch
m, n ,q = 10 , 7 ,5
lr = 0.05
X = torch.rand(m ,q)
W_true = torch.rand(q,n)
Y_true = X @ W_true + 0.1 * torch.randn(m,n) #0.1 Pour que le rapport signal bruit ne soit pas trop moche
W = torch.rand(q,n, requires_grad=True)
Y_pred = X @ W  # <=> torch.matmul(X,W) + B
for k in range(2000):
    Y_pred = X @ W 
    loss = 0.5 * ((Y_pred - Y_true)**2).sum() / m
    loss.backward() # Il faut définir une fonction objectif c'est cette fonction là dont on calcule le gradient et qui détermine la direction de la pente et ne pas oublier les parenthèses !
    with torch.no_grad():
        W -= lr * W.grad # use in place operation
        W.grad.zero_()
print(loss.item)
print(W_true - W) #Différence entre la vraie matrice et la matrice obtenue GG
