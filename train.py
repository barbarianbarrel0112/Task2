import uNet
from torch import optim
import sys
import numpy as np
import torch.nn as nn
import torch.utils.data
import torch
from readData import myData


sys.path.append(r"C:\Users\Administrator\AppData\Local\Programs\Python\Python310\Lib\site-packages")

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        return 1 - dice

rootDir = "datasetIMake"
trainSet = myData(rootDir, "train/image", "train/label")
valiSet = myData(rootDir, "vali/image", "vali/label")

trainLoader = torch.utils.data.DataLoader(dataset=trainSet, batch_size=2, shuffle=True)
valiLoader = torch.utils.data.DataLoader(dataset=valiSet, batch_size=2, shuffle=True)

testSize = len(valiSet)
model = uNet.UNet(n_channels=1, n_classes=1)
device = torch.device("cuda")
model = model.to(device)
lossF = DiceLoss()
lossF = lossF.to(device)
lR = 1e-3
optim = optim.Adam(params=model.parameters(), lr=lR, weight_decay=1e-5)

epochNum = 100
bestLoss = np.inf
bestAcc = 0.0
bestWeight = None
bestEpoch = 0
patience = 15
batchNum = len(trainLoader)
withoutDev = 0
for epoch in range(epochNum):
    print("epoch {}".format(epoch+1))
    eLoss = 0;
    batchIndex = 0;
    for data in trainLoader:
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        model.train()
        outputs = model(imgs)
        result_loss = lossF(outputs,targets)
        eLoss += result_loss
        optim.zero_grad()
        result_loss.backward()
        optim.step()
        batchIndex += 1
        if batchIndex % 4 == 0:
            print("{} / {}".format(batchIndex, len(trainLoader)))
            testLoss = 0
            testAcc = 0.0
            model.eval()
            with torch.no_grad():
                for data in valiLoader:
                    imgs, targets = data
                    imgs = imgs.to(device)
                    targets = targets.to(device)
                    outputs = model(imgs)
                    valiLoss = lossF(outputs, targets)
                    testLoss += valiLoss
            testLoss = testLoss / len(valiLoader)
            print("test loss: {}".format(testLoss))

            if bestLoss > testLoss:
                withoutDev = 0
                bestLoss = testLoss
                bestEpoch = epoch + 1
                bestWeight = model.state_dict()
                torch.save(bestWeight, "weights/bestWeight.pth")
            else:
                withoutDev += 1
            print("best loss: {}".format(bestLoss))
            print("patience: {}".format(patience - withoutDev))
            if withoutDev == patience:
                print("overFitting,exit.")
                print("best epoch: {}".format(bestEpoch))
                sys.exit()
    print("train loss: {}".format(eLoss))
    # weightName = 'weights/modelWeight' + str(epoch) + '.pth'
    # torch.save(model.state_dict(), weightName)