# BiLSTM_CRF_NER

This repo contains the source code and dataset for the basic BiLSTM-CRF model.

## Dependency package
The code has been tested using Python3.7 on CentOS and uses the following dependencies on a CPU and GPU.

Please install all dependencies using the command before running:

```
    $ pip install TensorFlow==1.15.2
    $ pip install Keras==2.3.1
    $ pip install git+https://www.github.com/keras-team/keras-contrib.git
```
## Content
- data
	- CDR corpus
- vocab
	- char.vocab: the vocabulary of char
	- cdr_label.vocab: the vocabulary of label
- src
	- NER_BiLSTM_CRF.py: main file
	- model.py: the model file
	- representation.py: convert input text to input vector
	- conlleval.pl: CONLL03 evluation script
	- processing_data.py

##Run
1. To run this code, you need to first download [the pretrained word embedding file.](https://drive.google.com/file/d/18KTxjkkTbntkWYLUl_brg34paVW0bNWQ/view?usp=sharing)
1. Run the "NER_BiLSTM_CRF.py" file

```
    $ python NER_BiLSTM_CRF.py
```