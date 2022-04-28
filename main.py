import argparse, os
from hmac import trans_36
import random

import torch
from Dataloader import getDataLoader

from utils import Logger, printModelSize
from Model import LeNet


#############################################################
# Hyper-parameters
#############################################################

def argParsing():

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=bool, default=True, help="train mode")
    ## Training policy
    parser.add_argument("--numEpoch", type=int, default=2000, help="num of epoch")
    parser.add_argument("--batchSize", type=int, default=128, help="input batch size")
    parser.add_argument("--lr", nargs="+", type=float, default=(1e-4), help="learing rate : Gen Dis")
    parser.add_argument("--manualSeed", type=int, default=1, help="manual seed")
    parser.add_argument("--pruneFreq", type=int, default="2", help="log term") 
    ## Environment
    parser.add_argument("--ngpu", type=int, default=1, help="number of gpus")
    parser.add_argument("--numWorkers", type=int, default=5, help="number of workers for dataloader")
    ## Data
    parser.add_argument("--savePath", type=str, default="./result/1/", help="path to save folder") 
    parser.add_argument("--logFreq", type=int, default="100", help="log term") 

    
    
    args = parser.parse_args()
    return args


def setEnv(args):

    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)

    try:
        if not os.path.exists(args.savePath):
            os.makedirs(args.savePath)
    except OSError:
        print("Error: Creating save folder. [" + args.savePath + "]")

    if torch.cuda.is_available() == False:
        args.ngpu = 0
    
    if args.ngpu == 1:
        args.device = torch.device("cuda")
    else :
        args.device = torch.device("cpu")


if __name__ == "__main__":

    args = argParsing()
    setEnv(args)
    logger = Logger(args.savePath)
    logger.log(str(args))

    model = LeNet(30).to(device=args.device)

    if args.train == True:
        logger.log("[[[Train]]] Train started..")
        printModelSize(model, logger)

        # Define data
        trainDataLoader = getDataLoader(True, args, logger)
        testDataLoader = getDataLoader(False, args, logger)

        # Define trainer
        #from Trainer import Tranier
        from PruningTrainer import Tranier
        trainer = Tranier(args=args, logger=logger)

        # Start training
        trainer.train(model, trainDataLoader, testDataLoader, 0.8)

    else :
        logger.log("[[[Evalution]]] Eval started..")
        printModelSize(model, logger)

        # Define data
        testDataLoader = getDataLoader(False, args, logger)
