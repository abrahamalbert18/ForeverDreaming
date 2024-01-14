import math
import torch
import torch.nn as nn
from torch.optim import AdamW
from tokenizers import Tokenizer
from torch.utils.data import DataLoader
from Dataset import ForeverDreamingDataset
from Models import ScriptWriter
from tqdm import tqdm
import argparse
from torch.utils.tensorboard import SummaryWriter

import os

torch.manual_seed(42)

maxSequenceLength = 100
batchSize = 10
numberOfEpochs = 10

tokenizer = Tokenizer.from_file(path="Tokenizer/ForeverDreaming/Vocab.json")
trainingDataset = ForeverDreamingDataset(splitType="train",
                                         maxSeqLength=maxSequenceLength)
validationDataset = ForeverDreamingDataset(splitType="val",
                                           maxSeqLength=maxSequenceLength)

lengthOfDatasets = {"train": len(trainingDataset),
                    "val": len(validationDataset)}
datasets = {"train": trainingDataset, "val": validationDataset}

def extractBatch(dataset, batchSize, item, maxSequenceLength, phase):
    """
    :param batchSize: Expected batchSize
    :param item: Index of the training/validation dataset row (episode
    transcription)
    :param maxSequenceLength: maximum number of tokens used during training
    :param phase: "train" or "val"
    :return: source, target (source shifted by 1 token)
    """
    data = dataset[item]
    if phase == "train":
        data = dataset[item]
    iterations = math.ceil(data.size(0) / batchSize)
    source = data[:, :maxSequenceLength - 1]  # Inputs
    target = data[:, 1:]  # Labels
    for i in range(iterations-1):
        yield (source[i * batchSize: (i+1) * batchSize, :],
               target[i * batchSize: (i+1) * batchSize, :])
    pass

# Model Config
contextLength = 512
numberOfHeads = 8
vocabSize = 30000
depth = 8
learningRate = 1e-3

modelConfig = {"contextLength": contextLength,
               "numberOfHeads": numberOfHeads,
               "vocabSize": vocabSize,
               "depth": depth,
               "maxSequenceLength": maxSequenceLength,
               "batchSize": batchSize}
cuda = torch.cuda.is_available()

model = ScriptWriter(contextLength=contextLength,
                         numberOfHeads=numberOfHeads,
                         vocabSize=vocabSize,
                         generate=False,
                         depth=depth)

# optimizer
softmax = nn.Softmax()
optimizer = AdamW(model.parameters(), lr=learningRate)
# optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)
# learning rate scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

device = torch.device("mps") # for mac
if cuda:
    device = torch.device("cuda:0") # for NVIDIA GPUs

# best metrics and parameters
bestEpoch = 0
bestEpochLoss = 15

modelName = f"ScriptWriter-->v1.pth.tar"
# Load from checkpoint
if os.path.exists(f"SavedModels/{modelName}"):
    checkpoint = torch.load(f"SavedModels/Encoder-->{modelName}",
                            map_location=device)
    model.load_state_dict(checkpoint["modelStateDict"])
    bestEpochLoss = checkpoint["bestEpochLoss"]
    learningRate = checkpoint["learningRate"]
    # optimizer.load_state_dict(checkpoint["optimizerStateDict"])
    del checkpoint

model.to(device)

writer = SummaryWriter(f"runs/{modelName}")

def start(bestEpoch, bestEpochLoss, modelConfig):
    # training and evaluation loop
    for epoch in tqdm(range(numberOfEpochs), desc="Epoch progress:", leave=False):
        print("-" * 40)
        print(f"Epoch {epoch + 1}:")
        # Setting phase
        for phase in ["train", "val"]:
            dataset = datasets[phase]
            print("-" * 40)
            print(f"Status : {phase}")
            print("-" * 40)
            if phase == "train":
                model.train()
            else:
                model.eval()

            epochLoss = 0
            iterations = 0
            # Loop 1 --> Iterate over all numberOfEpisodes
            for item in range(lengthOfDatasets[phase]):
                dataloader = extractBatch(dataset=dataset,batchSize=batchSize,
                                          item=item,
                                          maxSequenceLength=maxSequenceLength,
                                          phase=phase)
                iterationLoss = 0
                # Loop 2 --> Iterate within episode transcription
                for source, target in tqdm(iter(dataloader),
                                           desc="Iteration Progress",
                                           leave=False):

                    with torch.set_grad_enabled(phase == "train"):
                        source, target = source.to(device), target.to(device)
                        outputs, loss = model(source, target)

                        if phase == "train":
                            # Zero the gradients
                            optimizer.zero_grad()
                            # backpropgate the loss
                            loss.backward()
                            # update the weights
                            optimizer.step()

                    iterationLoss += loss.item()
                    iterations += 1
                epochLoss += (iterationLoss / iterations)
            """
             Epoch metrics
            """
            averageEpochLoss = epochLoss / (epoch + 1)
            if epoch % 1 == 0:
                print(f"{phase} loss = {averageEpochLoss:.4f}")
            writer.add_scalar(f"{phase.capitalize()} Loss/Epoch",
                              averageEpochLoss,
                              epoch + 1)
            if (averageEpochLoss < bestEpochLoss) and phase == "val":
                bestEpochLoss = averageEpochLoss
                bestEpoch = epoch
                if not os.path.exists(f"SavedModels/"):
                    os.mkdir("SavedModels")
                torch.save({"epoch":          epoch + 1,
                            "modelStateDict": model.state_dict(),
                            # "optimizerStateDict":optimizer.state_dict(),
                            "bestEpochLoss":  round(bestEpochLoss, 4),
                            "learningRate":   learningRate,
                            "modelConfig": modelConfig},
                           f"SavedModels/Encoder-->{modelName}")
            writer.close()
    print(f"Best loss: {round(bestEpochLoss, 4)} @ epoch #{bestEpoch + 1}")
    print(f"Best model saved.")
    pass

if __name__ == '__main__':
    # Model Config
    contextLength = 512
    numberOfHeads = 8
    vocabSize = 30000
    depth = 8
    learningRate = 1e-3
    batchSize = 10
    maxSequenceLength = 100
    modelConfig = {"contextLength":     contextLength,
                   "numberOfHeads":     numberOfHeads,
                   "vocabSize":         vocabSize,
                   "depth":             depth,
                   "maxSequenceLength": maxSequenceLength,
                   "batchSize":         batchSize}
    start(0,0, modelConfig)
    # dataloader = extractBatch(5, 0,10, "train")
    # for s, t in iter(dataloader):
    #     print(s, t)
    #     if s.numel() == 0:
    #         print(s)

