# Forever Dreaming
In this project I intend to train a language model using transformer 
architecture to write scripts. The dataset contains transcriptions of 
popular TV show episodes. For more information please visit
this [website](https://transcripts.foreverdreaming.org/). 

This work is intended for learning purposes.

The current Transformer model is trained with a vocabulary size of 30,000 
tokens and maximum sequence length of 100 tokens. It can run on Cloud TPUs 
as well as Apple Silicon processors.

##### Code to train the model
`python ModelTraining.py`

You can modify the model architecture using _Models.py_ file
Using _GenerateScripts.py_ you can generate text using the pretrained models.

Make sure you train your own tokenizer or use pretrained tokenizers.

Model Outputs:

* who is chandler? 
* carol : horrible man! 
* rachel : he can ’ t miss her. 
* phoebe : you alright? 
* rachel : i make you wish to hear me and pretend! 
* ross : hey,… how pheebs can you tell me? 
* phoebe : okay, you know what? 
* rachel : no, let me not say you.( laughs ) 
* phoebe : yeah! rachel : okay, well, let ’ s`

`
* sheldon cooper 
* : please. 
* leonard : okay. 
* cooper : as i got this tip, i ’ ve changed my mind. 
* sheldon : okay, did i ever imagine by the math of an apology? 
* sheldon : no. 
* howard : oh, look, look, we can teach them how things are usually learn to take them out. our heads in time pockets. 
* sheldon : what do you mean? 
* kripke : my brain was
