import torch
from Preprocess import Anime 
import torch
from torch import optim, nn
import visdom
import torchvision
from torch.utils.data import DataLoader
from Preprocess import Anime
from torchvision.models import resnet18,ResNet18_Weights
from utils import Flatten
import numpy as np

batchze = 32
lr = 1e-3
epochs = 10
device = torch.device('cuda')
torch.manual_seed(1234)

trained_model = resnet18(weights=ResNet18_Weights.DEFAULT)
model = nn.Sequential(*list(trained_model.children())[:-1],# [b,512,1,1]
                        Flatten(),
                        nn.Linear(512,4)
                        ).to(device)

model.load_state_dict(torch.load("best.mdl")) # load best state

db = Anime('..\\display', 512, 'display')
print(db.name2label)

loader = DataLoader(db,batch_size= 4)

# result = {}
# correct  = 0
# total = len(loader.dataset)
# for x , y in loader:
#     x, y  = x.to(device), y.to(device)
#     with torch.no_grad():
#         logits = model(x)
#         pred = logits.argmax(dim = 1)
#         # print(logits.shape)
#         print("pred = ", pred[0])
#         print("y = ", y[0])
#     correct += torch.eq(pred, y).sum().float().item()
# print(correct / total)

def evaluate(model, loader):
    correct = 0
    total = len(loader.dataset)
    for x , y in loader:
        x, y  = x.to(device), y.to(device)
        with torch.no_grad():
            logits = model(x)
            # print(logits)
            pred = logits.argmax(dim = 1)
            prob = logits.softmax(dim = 1).detach().cpu().numpy()
            print(np.round(prob,4))
            print(y)
        correct += torch.eq(pred, y).sum().float().item()
    return correct / total
evaluate(model, loader)