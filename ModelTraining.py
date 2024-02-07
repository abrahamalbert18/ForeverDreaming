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
#batchSize = 10
numberOfEpochs = 150

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
    iterations = max(2, math.ceil(data.size(0) / batchSize))
    source = data[:, :maxSequenceLength - 1]  # Inputs
    target = data[:, 1:]  # Labels
    for i in range(iterations-1):
        yield (source[i * batchSize: (i+1) * batchSize, :],
               target[i * batchSize: (i+1) * batchSize, :])
    pass

# Model Config
contextLength = 768
numberOfHeads = 8
vocabSize = 30000
depth = 4
learningRate = 1e-2

cuda = torch.cuda.is_available()
device = torch.device("mps") # for mac
if cuda:
    device = torch.device("cuda") # for NVIDIA GPUs

model = ScriptWriter(contextLength=contextLength,
                         numberOfHeads=numberOfHeads,
                         vocabSize=vocabSize,
                         generate=False,
                         depth=depth)
#model = torch.nn.DataParallel(model, device_ids=[0,1])
#model.to(device)

# optimizer
softmax = nn.Softmax()
#optimizer = AdamW(model.parameters(), lr=learningRate)
optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)
# learning rate scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

# best metrics and parameters
bestEpoch = 0
bestEpochLoss = 15
# Model Config
batchSize = 20
maxSequenceLength = 100
modelConfig = {"contextLength":     contextLength,
               "numberOfHeads":     numberOfHeads,
               "vocabSize":         vocabSize,
               "depth":             depth,
               "maxSequenceLength": maxSequenceLength,
               "batchSize":         batchSize}



modelName = f"Transformer-->ScriptWriter-->v1.pth.tar"
#modelName = "mini.pth.tar"
# Load from checkpoint
if os.path.exists(f"SavedModels/{modelName}"):
    checkpoint = torch.load(f"SavedModels/{modelName}",
                            map_location=device)
    model.load_state_dict(checkpoint["modelStateDict"])
    bestEpochLoss = checkpoint["bestEpochLoss"]
    learningRate = checkpoint["learningRate"]
    optimizer.load_state_dict(checkpoint["optimizerStateDict"])
    modelConfig = checkpoint["modelConfig"]
    print("Model and Optimizer loaded successfully")
    # del checkpoint

print(f"Previous model's best loss: {bestEpochLoss}")
model = torch.nn.DataParallel(model, device_ids=[0,1])
model.to(device)

writer = SummaryWriter(f"runs/{modelName}")

# training and evaluation loop
for epoch in tqdm(range(numberOfEpochs), desc="Epoch progress:", leave=False):
    tqdm.write("-" * 40)
    tqdm.write(f"Epoch {epoch + 1}:")
    # Setting phase
    for phase in ["train", "val"]:
        dataset = datasets[phase]
        #tqdm.write(f"Status : {phase}")
        #tqdm.write("-" * 40)
        if phase == "train":
            model.train()
        else:
            model.eval()

        epochLoss = 0
        # Loop 1 --> Iterate over all numberOfEpisodes
        for item in tqdm(range(lengthOfDatasets[phase]), desc="Episode "
                                                              "Progress",
                        unit=" Episode(s)", leave=False):
            dataloader = extractBatch(dataset=dataset,batchSize=batchSize,
                                      item=item,
                                      maxSequenceLength=maxSequenceLength,
                                      phase=phase)
            episodeLoss = 0
            iterations = 0
            # Loop 2 --> Iterate within episode transcription
            for source, target in tqdm(iter(dataloader),
                                       desc="MiniBatch Progress",
                                       leave=False,
                                       unit=" MiniBatch(es)"):
                if source.size(0) == 0:
                    break
                with torch.set_grad_enabled(phase == "train"):
                    source, target = source.to(device), target.to(device)
                    outputs, loss = model(source, target)
                    loss = loss.mean()
                    episodeLoss += loss.item()
                    iterations += 1
                    if phase == "train":
                        # Zero the gradients
                        optimizer.zero_grad()
                        # backpropgate the loss
                        loss.backward()
                        # Clipping the gradients
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                        # update the weights
                        optimizer.step()

            episodeLoss = episodeLoss / iterations
            #if item % 100 == 0:
            #    tqdm.write(f"{item}th episode loss: {episodeLoss}")
            epochLoss += episodeLoss
        """
         Epoch metrics
        """
        epochLoss = epochLoss / (lengthOfDatasets[phase])
        if epoch % 1 == 0:
            tqdm.write(f"{phase} loss = {epochLoss:.4f}")
        writer.add_scalar(f"{phase.capitalize()} Loss/Epoch",
                          epochLoss,
                          epoch + 1)
        if (epochLoss < bestEpochLoss) and phase == "val":
            bestEpochLoss = epochLoss
            bestEpoch = epoch
            if not os.path.exists(f"SavedModels/"):
                os.mkdir("SavedModels")
            torch.save({"epoch":          epoch + 1,
                        "modelStateDict": model.module.state_dict(),
                        "optimizerStateDict":optimizer.state_dict(),
                        "bestEpochLoss":  round(bestEpochLoss, 4),
                        "initialLearningRate":   learningRate,
                        "latestLearningRate":scheduler.get_last_lr(),
                        "modelConfig": modelConfig},
                       f"SavedModels/{modelName}")
            tqdm.write(f"Best loss: {round(bestEpochLoss, 4)} @ epoch "
                  f"#{bestEpoch + 1}")
            tqdm.write(f"Best model saved.")
        if (epoch % 15 == 0) and (epoch < 105):
           scheduler.step()
        writer.close()

# if __name__ == '__main__':
#     # Model Config
#     contextLength = 512
#     numberOfHeads = 8
#     vocabSize = 30000
#     depth = 8
#     learningRate = 1e-3
#     batchSize = 15
#     maxSequenceLength = 100
#     modelConfig = {"contextLength":     contextLength,
#                    "numberOfHeads":     numberOfHeads,
#                    "vocabSize":         vocabSize,
#                    "depth":             depth,
#                    "maxSequenceLength": maxSequenceLength,
#                    "batchSize":         batchSize}
#     print(f"Previous models best loss: {bestEpochLoss}")
#     start(0,bestEpochLoss, modelConfig)

