import torch
from tokenizers import Tokenizer
from Models import ScriptWriter
import argparse
import re

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--modelName",
                    default=f"Encoder-->ScriptWriter-->v6-->NoClipping.pth.tar")
parser.add_argument("-nv", "--cuda", default=False, type=bool)
parser.add_argument("-t", "--tokens", default=500, type=int)
parser.add_argument("-w", "--word", default="", type=str)
parser.add_argument("-v", "--vocabSize", default=30000, type=int)
parser.add_argument("-cl", "--contextLength", default=768, type=int)
parser.add_argument("-d", "--depth", default=4, type=int)
args = parser.parse_args()

modelName = args.modelName
cuda = args.cuda
numberOfTokens = args.tokens
firstWord = args.word
vocabSize = args.vocabSize
contextLength = args.contextLength
depth = args.depth


device = "mps"
device = "cpu"
if cuda:
    device = "cuda:0"

checkpoint = torch.load(f"SavedModels/{modelName}", map_location=device)


modelConfig = checkpoint["modelConfig"]
# Model Config
contextLength = modelConfig["contextLength"]
numberOfHeads = modelConfig["numberOfHeads"]
vocabSize = modelConfig["vocabSize"]
depth = modelConfig["depth"]
# learningRate = 1e-3
maxSequenceLength = modelConfig["maxSequenceLength"]


model = ScriptWriter(contextLength=contextLength,
                         numberOfHeads=numberOfHeads,
                         vocabSize=vocabSize,
                         generate=True,
                         depth=depth,
                        dropout=0)

#model = torch.nn.DataParallel(model)
model.load_state_dict(checkpoint["modelStateDict"])
model.eval()

softmax = torch.nn.Softmax(dim=-1)
# predictedWords = ""
print(f"{'-'*40}\n\n")
# newSource = torch.zeros(numberOfTokens // 100, maxSequenceLength - 1)
# newTarget = torch.zeros(numberOfTokens // 100, maxSequenceLength - 1)
memory = None

tokenizer = Tokenizer.from_file(path="Tokenizer/ForeverDreaming/Vocab.json")
sentence = f"{firstWord} "
def tokenizeSourceAndTarget(sentence):
    tokenizedSentence = tokenizer.encode(sequence=sentence)
    tokenizedTarget = tokenizedSentence.ids[1:]
    source = torch.tensor(tokenizedSentence.ids[:-1])
    target = torch.tensor(tokenizedTarget[:])
    return source, target

def generateText(source, target, tokenizer=tokenizer, model=model):
    for l in range(numberOfTokens):
        if source.size(-1) > (maxSequenceLength - 1):
            if l < maxSequenceLength:
                words = tokenizer.decode(target[:-1].short().tolist())
                memory = 1
            else:
                memory = 20
                words = tokenizer.decode(target[memory:-1].short().tolist())

            source = source[-1 * memory:]
            target = target[-1 * memory:]

        outputs = model(source.unsqueeze(0), target.unsqueeze(0)) # Encoder-Decoder
        nextTokenProbs = softmax(outputs[-1])
        predictions = torch.multinomial(nextTokenProbs,
                                        num_samples=1).to("cpu")
        source = torch.cat((source, predictions))
        target = torch.cat((target, predictions))

    # print(f"{'-'*40}")
    # Output Formatting
    template = r"\w* :"
    speakers = re.findall(template, words)
    content = re.split(template, words)
    formattedText = ""
    for i in range(len(speakers)):
        formattedText += (speakerFormatting(speakers[i]) +
                          contentFormatting(content[i + 1]) + "\n")

    print(words)
    return formattedText

def contentFormatting(content):
    content = content.replace("i ", "I ").strip()
    return content

def speakerFormatting(speaker):
    return speaker.capitalize() + " "

if __name__ == '__main__':
    source, target = tokenizeSourceAndTarget(sentence)
    text = generateText(source, target, tokenizer, model)
    print(text)

def generate(source):

    numberOfTokens = maxSequenceLength - 1 - source.size(-1)
    newSource = torch.zeros((1, numberOfTokens)).short()

    for j in range(5):
        for i in range(numberOfTokens):
            outputs = model(source.unsqueeze(0), target.unsqueeze(0)) # Encoder-Decoder
            nextTokenProbs = softmax(outputs[-1])
            predictions = torch.multinomial(nextTokenProbs,
                                            num_samples=1).to("cpu")
            source = torch.cat((source, predictions))
            target = torch.cat((target, predictions))
        newSource = torch.cat((newSource, source.unsqueeze(0)), dim=0)
        source = newSource[1:]
    words = tokenizer.decode(target[:-1].short().tolist())
    print(words)




# print(predictedWords)