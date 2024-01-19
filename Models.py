import torch
from torch import nn
import math

class ScriptWriter(nn.Module):
    def __init__(self, numberOfHeads=4, contextLength=32,
                 vocabSize=2000, generate=False, depth=6, dropout=0.2):
        super().__init__()
        self.vocabSize = vocabSize
        self.contextLength = contextLength
        self.numberOfHeads = numberOfHeads
        self.generate = generate
        self.depth = depth
        self.dropout = dropout
        self.wordEmbedding = nn.Embedding(self.vocabSize, self.contextLength)
        self.positionEmbedding = PositionalEncoding(self.contextLength)
        self.transformerNetwork = nn.Transformer(nhead=self.numberOfHeads,
                                                 batch_first=True,
                                                 d_model=self.contextLength,
                                                 num_encoder_layers=self.depth,
                                                 num_decoder_layers=self.depth)
        # self.encoderLayer = nn.TransformerEncoderLayer(d_model=self.contextLength,
        #                                                 nhead=self.numberOfHeads,
        #                                                batch_first=True,
        #                                                dropout=self.dropout)
        # self.decoderLayer = nn.TransformerDecoderLayer(d_model=self.contextLength,
        #                                                 nhead=self.numberOfHeads,
        #                                                batch_first=True,
        #                                                dropout=self.dropout)
        # self.encoderNetwork = nn.TransformerEncoder(encoder_layer=self.encoderLayer,
        #                                             num_layers=self.depth)
        # self.decoderNetwork = nn.TransformerDecoder(decoder_layer=self.decoderLayer,
        #                                             num_layers=self.depth)

        self.criterion = nn.CrossEntropyLoss(ignore_index=3,
                                                 reduction="mean")

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.layerNorm = nn.LayerNorm(self.contextLength)
        self.predictionLayer = nn.Linear(self.contextLength,
                                self.vocabSize)

    def forward(self, encoderInputs, decoderInputs):
        source = self.layerNorm(self.wordEmbedding(encoderInputs.long()))
        source = self.positionEmbedding(source)
        target = self.layerNorm(self.wordEmbedding(decoderInputs.long()))
        target = self.positionEmbedding(target)
        outputs = self.transformerNetwork(src=source, tgt=target)
        # outputs = self.encoderNetwork(src=source)
        # outputs = self.decoderNetwork(tgt=source, memory=target)
        # outputs = self.layerNorm(outputs)
        outputs = self.predictionLayer(outputs) # B, T, VocabSize
        outputs = outputs.view(-1, outputs.size(-1)) # B * T, VocabSize
        if self.generate:
            return outputs

        loss = self.criterion(outputs, decoderInputs.long().view(-1))
        return outputs, loss

class PositionalEncoding(nn.Module):
    def __init__(self, contextLength, maxSequenceLength=500,
                 dropout=0.1):
        super().__init__()
        # self.dropout = nn.Dropout(p=dropout)

        positionalEmbedding = torch.zeros(maxSequenceLength, contextLength)
        #PositionalEmbedding
        position = torch.arange(maxSequenceLength).unsqueeze(1)
        # According to "Attention is All You Need" Paper
        scaleFactor = torch.exp(torch.arange(0, contextLength, 2)
                                * (-math.log(10000.0) / contextLength))
        positionalEmbedding[:, 0::2] = torch.sin(position * scaleFactor)
        positionalEmbedding[:, 1::2] = torch.cos(position * scaleFactor)
        positionalEmbedding = positionalEmbedding.unsqueeze(0)
        self.register_buffer("positionalEmbedding", positionalEmbedding)

    def forward(self, inputs):
        # @torch.no_grad
        return inputs + self.positionalEmbedding[:, :inputs.shape[1],
                        :].requires_grad_(False)
        # return self.dropout(x)