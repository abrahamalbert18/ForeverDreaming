import torch
from tokenizers import Tokenizer
from Models import ScriptWriter
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--modelName",
                    default=f"Transformer-->ScriptWriter-->v1.pth.tar")
parser.add_argument("-nv", "--cuda", default=False, type=bool)
parser.add_argument("-t", "--tokens", default=500, type=int)
parser.add_argument("-w", "--word", default="", type=str)
parser.add_argument("-v", "--vocabSize", default=30000, type=int)
parser.add_argument("-cl", "--contextLength", default=512, type=int)
parser.add_argument("-d", "--depth", default=8, type=int)
args = parser.parse_args()

modelName = args.modelName
cuda = args.cuda
numberOfTokens = args.tokens
firstWord = args.word
vocabSize = args.vocabSize
contextLength = args.contextLength
depth = args.depth

tokenizer = Tokenizer.from_file(path="Tokenizer/ForeverDreaming/Vocab.json")
sentence = f"{firstWord} "
tokenizedSentence = tokenizer.encode(sequence=sentence)
tokenizedTarget = tokenizedSentence.ids[1:]
source = torch.tensor(tokenizedSentence.ids[:-1])
target = torch.tensor(tokenizedTarget[:])

if not cuda:
   checkpoint = torch.load(f"SavedModels/{modelName}", map_location="mps")
else:
    checkpoint = torch.load(f"SavedModels/{modelName}", map_location="cuda")

modelConfig = checkpoint["modelConfig"]
# Model Config
contextLength = modelConfig["contextLength"]
numberOfHeads = modelConfig["numberOfHeads"]
vocabSize = modelConfig["vocabSize"]
depth = modelConfig["depth"]
# learningRate = 1e-3
maxSequenceLength = modelConfig["maxSequenceLength"]

device = "mps"
if cuda:
    device = "cuda:0"

model = ScriptWriter(contextLength=contextLength,
                         numberOfHeads=numberOfHeads,
                         vocabSize=vocabSize,
                         generate=True,
                         depth=depth,
                        dropout=0)

#model = torch.nn.DataParallel(model)
model.load_state_dict(checkpoint["modelStateDict"])
model.eval()

print(f"{'-'*40}\n\n")
for l in range(numberOfTokens):
    if source.size(-1) > (maxSequenceLength - 1):
        words = tokenizer.decode(target[:-1].short().tolist())
        # predictedWords.append(words)
        print(words)
        source = source[-1:]
        target = target[-1:]

    outputs = model(source.unsqueeze(0), target.unsqueeze(0)) # Encoder-Decoder
    nextTokenProbs = outputs[-1].softmax(dim=-1)
    #print("Actual Ouptuts:")
    #print(tokenizer.decode((torch.multinomial(outputs.softmax(dim=-1), num_samples=1)).reshape(-1).tolist()))
    #print()
    predictions = torch.multinomial(nextTokenProbs,
                                    num_samples=1).to("cpu")
    predictions = torch.argmax(nextTokenProbs).to("cpu").unsqueeze(0)

    source = torch.cat((source, predictions))
    target = torch.cat((target, predictions))

print(f"{'-'*40}")
