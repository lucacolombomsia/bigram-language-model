# Bigram Language Models applied to the task of language identification

*MSiA 490: Text Analytics*   
**Developer: Luca Colombo**  

This project uses different bigram-based statistical language models to accomplish language identification, which is the problem of taking as input a text in an unknown language and determine what language it is written in. In particular, models were trained to predict whether a line from a test corpus is one of three languages: English, French, or Italian. 


## Repository structure
This repository contains:
* `support.py`: a Python script that contains a series of functions that are common across all bigram-based language models.
* `letterLangId.py`: a letter bigram model with add-one smoothing.
* `wordLangId.py`: a word bigram model with add-one smoothing.
* `wordLangId2.py`: a word bigram model with Good-Turing smoothing.
* a series of flat files: these are the training corpora (one per language), the test corpus and the ground truth file (to compute the accuracy of the three models).


## Suggested steps to run the program 
1. Clone the repository.
2. Run any one of the three models either from the command line (for example, for the letter bigram model, execute `python letterLangId.py`) or using an IDE like Spyder.
3. An output file will be created; its name will match the name of the model you just trained and used to score the test corpus. The accuracy of the model on the test corpus will be written to the console.


## Design decisions
All lines in the corpora were cleaned by removing punctuation, casing, and extra whitespaces to focus the lanugage model on the relevant text. Additionally, `<b>` and `<e>` tokens were added to each line (for word-based models) and to each word (for letter-based models) to account for the relative position of a letter/word in a word/sentence. Model specific decisions are explained using inline comments in the code and in the following sections.

## Question 1
The letter bigram model cannot be implemented without smoothing. Unknown bigrams have 0 count, which means that any sentence with at least one unknown would have a predicted probability of 0. Given that most sentences in the test corpus contain at least one bigram that was not seen in the training corpus, this would make the model basically useless. Add-one smoothing is an effective way to solve the issues mentioned above, as it removes 0 counts from the data. I therefore decided to implement add-one smoothing, which gave me an accuracy of 98.7% (296/300). The strong performance of this model with add-one smoothing shows that there is no need for a more advanced algorithm. Also, changing thresholds did not lead to performance improvement; given that runtime is not an issue for this small model, I opted for a threshold of 0.

## Question 2
The word bigram model cannot be implemented without smoothing, for the same exact reasons explained in the previous section. Inevitably, unknown bigrams will be found in the test corpus; since these bigrams have 0 count, the predicted probability of the entire sentence would be 0. Moreover, bigrams that begin with a word that was not observed in the training corpus would cause problems, as to compute the conditional probability we would have to divide by 0 (this problem was less of a concern for the letter bigram, as it's more likely that we will observe all letters in training). Add-one smoothing is an effective way to solve the issues mentioned above, as it removes 0 counts from the data. I therefore decided to implement add-one smoothing, which gave me an accuracy of 99.7% (299/300). The strong performance of this model with add-one smoothing shows that there is no need for a more advanced algorithm. Since the training corpora are small, a threshold larger than 0 would lead to substantial vocabulary reduction. This observation, together with the fact that runtime is not an issue for this small model, led me to opt for a threshold of 0. Moreover, the source of the training data seems to be transcripts of either the EU parliament or the EU commission; hence typos should not be a concern and we can be even more confident when using a threshold of 0.

## Question 3
In the third model, instead of using add-one smoothing, I implemented Good-Turing smoothing to the word bigram model. Good-Turing is a good smoothing technique if and only if for any N(x) we have that N(x+1) > 0; otherwise, the smoothed count would be equal to zero. For this reason, I searched the list of Nx's to find the last Nx s.t. N(x+1) > 0 and decided to apply Good-Turing smoothing only to tokens associated to counts below this threshold. After applying Good-Turing smoothing, I obtained a perfect accuracy of 300/300. However, the model runs for a significantly longer amount of time than the add-one smoothing model and the idea behind Good-Turing is definitely harder to explain to a broad audience that includes people who are not familiar with text analytics.

## Conclusion: advantages and disadvantages of the three models
Each model has its own advantages and disadvantages. If I had to give a single recommendation, based on the results presented above, I would opt for the add-one smoothed word bigram model. I think that for the task at hand (and given the training and test corpora) it proved to be the best model: it runs fast, it is interpretable and it is basically just as accurate as its Good-Turing smoothed counterpart.    
However, if with different training and test corpora Good-Turing was to provide a substantially greater accuracy, I would favor that model. The letter bigram model was marginally less accurate than the other two models: however, it is a very simple model, it runs very fast and only requires us to keep track of an alphabet and not a vocabulary of words. If faced by substantial runtime problems, opting for this model could be beneficial.
