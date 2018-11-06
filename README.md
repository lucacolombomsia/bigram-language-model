# Bigram Language Models applied to the task of language identification

*MSiA 490: Text Analytics*   
**Developer: Luca Colombo**  

This program uses different bigram-based statistical language models to accomplish language identification, which is the problem of taking as input a text in an unknown language and determine what language it is written in. The program was trained to recognize three languages: English, French and Italian.


## Repository structure
This repository contains:
* `support.py`: a Python script that contains a series of functions that are common across all bigram-based language models
* `letterLangId.py`: a letter bigram model with add-one smoothing
* `wordLangId.py`: a word bigram model with add-one smoothing
* `wordLangId2.py`: a word bigram model with Good-Turing smoothing
* a series of flat files: these are the training files (one per language), the test file and the ground truth file (to compute the accuracy of the three models).


## Suggested steps to run the program 

1. Clone the repository.
2. Run any one of the three models either from the command line (for example, for the letter bigram model, execute `python letterLangId.py`) or using an IDE like Spyder.
3. An output file will be created; its name will match the name of the model you just trained and used to score the test data. The accuracy of the model on the test data will be written to the console.
