from models import ViT, ViR
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from draw_loss_curve import draw_result

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



if __name__ == '__main__':

    torch.manual_seed(0) # You should have 48% acc in epoch 1.
    layer_cnt = 8
    patch_pix = 4

    model = ViT( #ViT
        image_pix = 32,
        patch_pix = patch_pix,
        class_cnt = 10,
        layer_cnt = layer_cnt,
    )

    device = "cuda:0"
    batch_size = 8
    lr = 5e-6 # will iterate through 5e-6 ~ MAX, where MAX = 5e-4, MAX will decay by half every epoch.
    epochs = 10

    print(f"model has {count_parameters(model)} paramters")


    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    data_train = DataLoader(trainset, batch_size= batch_size,
                                            shuffle=True, num_workers=3)

    testset = datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    data_test = DataLoader(testset, batch_size= batch_size,
                                            shuffle=False, num_workers=3)

    #
    #model = torch.load("model.dat")

    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay = 1e-9, betas = [0.5, 0.99])
    scheduler = torch.optim.lr_scheduler.CyclicLR(opt, base_lr= lr, max_lr= lr*100, step_size_up = len(trainset)//(2 * batch_size), mode="triangular2", cycle_momentum= False) 
    criterion = nn.CrossEntropyLoss()
    train_losses = []
    test_losses = []
    accs = []
    

    for epoch in range(1, epochs + 1):
        acc = 0
        train_loss = 0
        test_loss = 0
        train_cnt = 0
        test_cnt = 0
        model.train()
        pbar = tqdm(data_train)

        for x, y in pbar:
            #print(x.shape)
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)  
            opt.zero_grad()
            loss = criterion(y_pred, y)
            loss.backward()
            opt.step()
            train_loss += loss.item()*x.shape[0]
            train_cnt += x.shape[0]
            pbar.set_description(f"Loss : {train_loss/train_cnt:.4f}, lr : {scheduler.get_last_lr()[0]:.3e}")
            scheduler.step()
            
            
        model.eval()
        with torch.no_grad():
            for x, y in data_test:
                x = x.to(device)
                y = y.to(device)
                
                y_pred = model(x)
                loss = criterion(y_pred, y)
                y_argmax = y_pred.argmax(dim = 1)
                acc = acc + (y == y_argmax).sum().item()
                test_cnt += x.shape[0]
                test_loss += loss.item() * x.shape[0]
                
            
            print(f'epoch {epoch} : Train loss : {train_loss/train_cnt:.4f}, Test loss : {test_loss/test_cnt:.4f} Test acc : {acc/test_cnt:.4f}')

        train_losses.append(train_loss/train_cnt)
        test_losses.append(test_loss/test_cnt)
        accs.append(100 * acc/test_cnt)

        torch.save(model, "model.dat")

    
    draw_result(epochs, train_losses, test_losses, accs, f"ViT_ly{layer_cnt}_px{patch_pix}")
    
    

