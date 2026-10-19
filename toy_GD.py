import torch
x = torch.tensor(5. ,requires_grad = True)
y = torch.tensor(7. ,requires_grad = True)
lr = 0.009
for k in range(10):
    z = 100 * x**2 + y**2 #z = 2549
    z.backward() #Ici pas besoin de fonction de perte on cherche directement à minimiser f(x,y)=100x² + y
    with torch.no_grad():
        x -= lr * x.grad # use in place operation
        x.grad.zero_()
        y -= lr * y.grad
        y.grad.zero_()
print(z) 
#On retrouve alors que L=200 et mu = 2 => converge ssi alpha < 2/L = 0.01
# Optimum alpha = 2/(L+mu)

