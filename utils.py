import numpy as np
import os
import torch
from PIL import Image

class Logger():
    def __init__(self, path):
        self.logFile = open(path+"log.txt", "w")
    def __del__(self):
        self.logFile.close()

    def log(self, logStr):
        print(logStr)
        self.logFile.write(logStr+"\n")
        self.logFile.flush()


def weightsInit(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)


def printModelSize(model, logger):
    numParams = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.log("Total number of trainable parameters : %d (%.3fGB)" %(numParams, float(numParams)*8/pow(2,30)))


def listAllImg(dataPath):
    filePaths = []
    for root, dir, files in os.walk(dataPath):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".png"):
                filePath = os.path.join(root, file)
                filePaths.append(filePath)
            
    return filePaths

def saveImage(args, epoch, i, imgs, interval=150):
    
    if i % interval != 0:
        return
        
    imgs = [img for img in imgs if img != None]

    for idx in range(len(imgs)):
        imgs[idx] = np.transpose(np.float32(imgs[idx].to("cpu").detach().numpy()[0])*255, (1,2,0))

    img = combine(imgs)
    img.save(args.savePath+"img_%d_%d.png"%(epoch,i))

def combine(imgs):

    for i in range(len(imgs)):
        imgs[i] = Image.fromarray(np.uint8(imgs[i]))

    widths, heights = zip(*(i.size for i in imgs))
    totalWidth = sum(widths)
    totalHeight = max(heights)

    new_img = Image.new("RGB", (totalWidth, totalHeight))
    offset = 0
    for img in imgs:
        new_img.paste(img, (offset, 0))
        offset += img.size[0]

    return new_img