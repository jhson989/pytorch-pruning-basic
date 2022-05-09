from audioop import avg
import torch
import torch.optim as optim
import torch.nn.utils.prune as prune
from torch import nn

class Tranier:

    def __init__(self, args, logger):

        ### Arguments
        self.args = args
        self.logger = logger

        ### Train Policy
        # criterion
        self.crit = nn.CrossEntropyLoss().to(self.args.device)
        self.pruned = False
        self.bestLoss = 2.0

    def train(self, model, dataLoader, evalDataLoader=None):

        print("Prune trainer")

        ### Data
        self.dataLoader = dataLoader
        self.evalDataLoader = evalDataLoader

        ### Model
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)



        ### Training iteration
        for epoch in range(self.args.numEpoch):

            ### Prune
            if epoch % self.args.pruneFreq == 0:   
                self.prune()

            ### Train
            avgLoss = 0.0
            self.model.train()
            for idx, (img, gt) in enumerate(self.dataLoader):

                ### learning
                img, gt = img.to(self.args.device), gt.to(self.args.device)
                self.optimizer.zero_grad()
                pred = self.model(img)
                loss = self.crit(pred, gt)
                loss.backward()
                self.optimizer.step()

                ### Logging
                avgLoss = avgLoss + loss.item()
                if idx % self.args.logFreq == 0 and idx != 0: 
                    self.logger.log("[[%4d/%4d] [%4d/%4d]] loss CE(%.3f)" 
                            % (epoch, self.args.numEpoch, idx, len(self.dataLoader), avgLoss/self.args.logFreq))
                    avgLoss = 0.0


            ### Eval
            if self.evalDataLoader is not None :
                self.eval(self.evalDataLoader, epoch)
                


    def eval(self, evalDataLoader, epoch):
        
        ### Eval
        self.model.eval()
        with torch.no_grad():
            avgLoss = 0.0
            for idx, (img, gt) in enumerate(evalDataLoader):

                ### Forward
                img, gt = img.to(self.args.device), gt.to(self.args.device)
                pred = self.model(img)
                loss = self.crit(pred, gt)

                avgLoss = avgLoss + loss.item()

            ### Logging
            avgLoss = avgLoss/len(evalDataLoader)
            self.logger.log("Eval loss : CE(%.3f)" 
                    % (avgLoss))

            if avgLoss < self.bestLoss :
                self.bestLoss = avgLoss
                self.save(epoch)


    def prune(self):
        
        if self.pruned == True :
            self.remove()

        self.pruned = True
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
                prune.l1_unstructured(module, name='weight', amount=self.args.pruneAmount)
                prune.l1_unstructured(module, name='bias', amount=self.args.pruneAmount)


    def remove(self):
        if self.pruned == False:
            return
            
        self.pruned = False
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
                prune.remove(module, 'weight')
                prune.remove(module, 'bias')


    def save(self, numEpoch):
        filename = self.args.savePath + "%d.pth" % numEpoch
        torch.save(self.model.state_dict(), filename)

    def load(self, filename):
        filename = self.args.savePath + filename
        torch.load_state_dict(torch.load(filename))