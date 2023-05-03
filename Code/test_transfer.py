import torch
from torch import optim, nn
import visdom
import torchvision
from torch.utils.data import DataLoader
from Preprocess import Anime
from torchvision.models import resnet18,ResNet18_Weights
from utils import Flatten



batchze = 32
lr = 1e-3
epochs = 20
device = torch.device('cuda')
torch.manual_seed(1234)

# load data 
train_db = Anime('..\\Images', 512, 'train')
val_db = Anime('..\\Images', 512, 'val')
test_db = Anime('..\\Images', 512, 'test')


train_loader = DataLoader(train_db,batch_size= batchze, shuffle= True)
val_loader = DataLoader(val_db,batch_size= batchze)
test_loader = DataLoader(test_db,batch_size= batchze)

def evaluate(model, loader):
    correct = 0
    total = len(loader.dataset)
    for x , y in loader:
        x, y  = x.to(device), y.to(device)
        with torch.no_grad():
            logits = model(x)
            pred = logits.argmax(dim = 1)
        correct += torch.eq(pred, y).sum().float().item()
    return correct / total

viz = visdom.Visdom()
def main():
    # model = ResNet18(4).to(device)
    trained_model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model = nn.Sequential(*list(trained_model.children())[:-1],# [b,512,1,1]
                          Flatten(),
                          nn.Linear(512,4)
                          ).to(device)
    optimizer = optim.Adam(model.parameters(),lr = lr)
    criteon = nn.CrossEntropyLoss()

    best_acc, best_epoch = 0, 0
    global_step = 0
    viz.line([0], [-1], win='loss', opts=dict(title='loss'))
    viz.line([0], [-1], win='val_acc', opts=dict(title='val_acc'))
    for epoch in range(epochs):
        for step, (x,y) in enumerate(train_loader):
            # x: [b,3,512,512] y:[b]
            x, y  = x.to(device), y.to(device)

            logits = model(x)
            loss = criteon(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            viz.line([loss.item()], [global_step], win='loss', update='append')
            global_step += 1
        if epoch % 1 == 0:
            val_acc = evaluate(model, val_loader)
            if val_acc > best_acc:
                best_epoch = epoch
                best_acc = val_acc
                torch.save(model.state_dict(),'best.mdl')
                viz.line([val_acc], [global_step], win='val_acc', update='append')
    print("best acc:", best_acc, "best epoch:", best_epoch)
    model.load_state_dict(torch.load("best.mdl")) # load best state

    print("loaded from checkpoint! ")

    test_acc = evaluate(model, test_loader)
    print("test acc:", test_acc)

main()