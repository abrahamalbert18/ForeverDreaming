# Forever Dreaming
In this project I intend to train a language model using transformer 
architecture to write scripts. The dataset contains transcriptions of 
popular TV show episodes. For more information please visit
this [website](https://transcripts.foreverdreaming.org/). 

This work is intended for learning purposes.

The current Transformer model is trained with a vocabulary size of 30,000 
tokens and maximum sequence length of 500 tokens. It can run on Cloud TPUs as well as Apple Silicon processors.

##### Code to train the model
`python ModelTraining.py`

You can modify the model architecture using _Models.py_ file
Using _GenerateScripts.py_ you can generate text using the pretrained models.

Make sure you train your own tokenizer or use pretrained tokenizers.

Training Statistics:
A batch size of 15 with maximum tokens of 500 per example requires you at 
least 14 GB of GPU RAM. It takes at least 2 hours to finish 1 epoch of the 
dataset on NVIDIA TESLA T4 GPU (Freely available on Google Colab).