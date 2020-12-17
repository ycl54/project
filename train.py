import torch
from dataset import Data
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from autoencoder import AEModel
from loss_ssim import SSIM
import copy
from torch.autograd import Variable
from torch.nn import MSELoss
from PIL import Image
def print_training_loss_summary(loss, total_steps, current_epoch, n_epochs, n_batches, print_every=10):
    # prints loss at the start of the epoch, then every 10(print_every) steps taken by the optimizer
    steps_this_epoch = (total_steps % n_batches)

    if (steps_this_epoch == 1 or steps_this_epoch % print_every == 0):
        print('Epoch [{}/{}], Iteration [{}/{}], Loss: {:.4f}'
              .format(current_epoch, n_epochs, steps_this_epoch, n_batches, loss))

transform1 = transforms.Compose([transforms.RandomRotation(degrees=5),
                                transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
                                transforms.Pad(padding=0, padding_mode='edge'),
                                transforms.Grayscale(1),
                                transforms.Resize((928, 928)),
                                transforms.ToTensor()])
transform2 = transforms.Compose([transforms.Resize((928, 928)),
                                 transforms.Grayscale(1)])
train_ds = Data('train', transform1)
train_dl = DataLoader(train_ds, batch_size=8, shuffle=True)

valid_ds = Data('valid', transform2)
valid_dl = DataLoader(valid_ds, batch_size=8, shuffle=True)

model = torch.load("trained_model.pt")
input = iter(train_dl).next().cuda()
model.eval()
with torch.no_grad():
    pred = model(input)
    print(np.uint8(pred[0].cpu().numpy()*255).reshape(928,928))
    im = Image.fromarray(np.uint8(pred[0].cpu().numpy()*255).reshape(928,928))
    im.save("res.jpg")
'''

model = AEModel("grayscale")

model = model.cuda()
num_epochs = 5
criterion = MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr = 0.01)
best_loss = float("inf")
total_steps = 0
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for batch in train_dl:
        image = Variable(batch.float())
        image_copy = copy.deepcopy(image)
        image = image.cuda()
        image_copy = image_copy.cuda()

        pred = model(image)
        loss = criterion(pred, image_copy)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        total_steps += 1
        print_training_loss_summary(loss.item(),total_steps, epoch+1, num_epochs, len(train_dl))
    valid_loss = 0.0
    model.eval()
    with torch.no_grad():
        for batch in valid_dl:
            image = Variable(batch.float())
            image_copy = copy.deepcopy(image)
            image = image.cuda()
            image_copy = image_copy.cuda()

            pred = model(image)
            loss = criterion(pred, image_copy)
            valid_loss += loss.item()

    if(best_loss > valid_loss):
        best_loss = valid_loss
        torch.save(model, "trained_model.pt")
    print("train_loss,{t}; valid_loss,{v}".format(t=train_loss, v=valid_loss))

'''
