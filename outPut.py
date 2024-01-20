import os

import numpy as np
import torch.utils.data
import torch
from PIL import Image
import uNet
from readData import myData4Test

rootDir = "dataset"
testDir = "test/image"
testSet = myData4Test(rootDir, testDir)

testLoader = torch.utils.data.DataLoader(dataset= testSet, batch_size= 2, shuffle= False)

model = uNet.UNet(n_channels=1, n_classes=1)

model.load_state_dict(torch.load("weights/bestWeight.pth"))
device = torch.device("cuda")
model = model.to(device)
testSize = len(testSet)
testLoss = 0
outDir = "dataset/test/label"
if not os.path.exists(outDir):
    os.mkdir(outDir)
i = 0
j = 0
model.eval()
with torch.no_grad():
    for data in testLoader:
        imgs, names = data
        imgs = imgs.to(device)
        outputs = model(imgs)
        for i in range(2):
            imgTensor = outputs[i]
            imgTensor[imgTensor > 0] = 255
            imgTensor[imgTensor <= 0] = 0
            imgAbsName = os.path.join(outDir, names[i])
            imgNP = np.array(imgTensor.data.cpu()[0],dtype=np.uint8)
            patch = Image.fromarray(imgNP)
            patch.save(imgAbsName)
        j += 1
        print("{} / {}".format(j, len(testLoader)))
