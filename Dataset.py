import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from tokenizers import Tokenizer
import pandas as pd

class ForeverDreamingDataset(Dataset):
    def __init__(self,
                 filename="Data/train-00000-of-00001"
                          "-5d885a32fc0cda9b.parquet",
                 splitType="train", maxSeqLength=100):
        super().__init__()
        self.filename = filename
        self.df = pd.read_parquet(self.filename)
        self.tokenizer = Tokenizer.from_file(
                path="Tokenizer/ForeverDreaming/Vocab.json")
        self.trainSplits, self.valSplits = self.generateSplits()
        self.splitType = splitType
        self.maxSeqLength = maxSeqLength

    def loadData(self, item):
        text = self.df["TEXT"][item.item()] #Rows of Texts
        text = text.strip().splitlines()
        return text

    def generateSplits(self):
        torch.manual_seed(42)
        lengthOfTheDataset = len(self.df)
        randomIndices = torch.randint(0, lengthOfTheDataset,
                                      (lengthOfTheDataset,))
        splitValue = round(0.85 * lengthOfTheDataset)
        trainIndices = randomIndices[: splitValue]
        valIndices = randomIndices[splitValue:]
        return trainIndices, valIndices

    def __getitem__(self, item):
        if self.splitType == "train":
            item = self.trainSplits[item]
        else:
            item = self.valSplits[item]
        sentences = self.loadData(item)

        tokenizedSentences = []
        for sentence in sentences:
            tokenizedSentence = self.tokenizer.encode(sequence=sentence)
            if len(tokenizedSentence.ids) > 2:
                tokenizedSentences.append(torch.tensor(tokenizedSentence.ids))
        tokenizedSentences = torch.cat(tokenizedSentences) # (Sequence,)
        numberOfTokens = tokenizedSentences.size(0)
        tokensToPad = (self.maxSeqLength - numberOfTokens % self.maxSeqLength)
        paddingTokens = 3 * torch.ones(tokensToPad, dtype=torch.long)
        if paddingTokens.numel() != 0:
            data = torch.cat((tokenizedSentences, paddingTokens))
        else:
            data = tokenizedSentences.clone()
        data = data.view(-1, self.maxSeqLength)  # B * SeqLen (For Mac)
        return data

    def __len__(self):
        if self.splitType == "train":
            return len(self.trainSplits)
        return len(self.valSplits)

if __name__ == '__main__':
    fd = ForeverDreamingDataset(splitType="train")
    t1 = fd[1]
    print(t1)