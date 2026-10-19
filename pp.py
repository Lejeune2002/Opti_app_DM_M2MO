import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

"""
data = [[1, 2 ],[3, 4] ,[5, 6]] #Initlialisation tenseur par un array
tensor = torch.rand(3,4) #Crée une matric 3x4 de floats avec des valeurs alééatoires
#Un tenseur c'est juste un array qui tourne sur GPU
"""

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten() #Applatit une matrice en un vecteur
        self.linear_relu_stack = nn.Sequential(#Permet d'enchaîner les couches
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            #Ici on a enchainé un réseau à 5 couches : nn.Linear (in => Dimension d'entrée, out => Dimension de sortie ) et les RELU(z)=max(z,0) => Non linéarité du modèle
        )

    def forward(self, x): #Fait passer x dans le réseau 
        x = self.flatten(x)  #On écrase comme tout à l'heure (28,28) => (1,784)
        logits = self.linear_relu_stack(x) #On passe dans le réseaux initialisé
        return logits #On retourne la valeur obtenue
model = NeuralNetwork().to(device)
#print(model)

X = torch.rand(1, 28, 28, device=device)
logits = model(X) #On appelle implicitement forward ici
pred_probab = nn.Softmax(dim=1)(logits) #Softmax renvoie des probas [0,1] à la place (softmax(Z_i)=exp(Z_i)/sum(exp(Z_i))  )
y_pred = pred_probab.argmax(1) #La classe prédite est la plus grande valeur de logit
print(f"Predicted class: {y_pred}")