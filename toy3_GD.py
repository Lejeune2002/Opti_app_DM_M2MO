#régression linéaire classique
import torch 
n=100
lr=0.05
B_true = torch.tensor([2,1])
d_true = torch.tensor(1*[n])
X = torch.rand(n,2)
Y_true = X @ B_true + d_true + 0.1*torch.randn(n,1)
B = torch.rand(2,1 ,requires_grad=True)
d = torch.rand(n,1 , requires_grad=True)
for k in range (20):
    Y_pred = X @ B + d
    loss =  (((Y_pred - Y_true)**2).sum())** 0.5 / n
    loss.backward()
    with torch.no_grad():
        B -= lr*B.grad
        d -= lr*d.grad
        B.grad.zero_()
        d.grad.zero_()
print(loss.item)
print (B-B_true)
print(d-d_true)


    

