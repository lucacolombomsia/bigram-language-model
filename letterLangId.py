from support import read_files, text_preprocess, write_out, compute_performance
import numpy as np
import operator

def make_letter_key(one_word, i):
    #this function, if called while iterating through the letters of a word,
    #returns the letter bigram
    #special characters <b> and <e> are used to identify beginning and
    #end of word respectively
    #two letters in bigram are separated by comma
    #'a,b' indicates that a is followed by b
    #adding comma makes it much easier to extract unigram for conditional proba
    #this is because with special characters, we do not fixed byte length
    if i == 0:
        key = '<b>,' + one_word[i]
    elif i == len(one_word):
        key = one_word[i-1] + ',<e>'
    else:
        key = ','.join(one_word[i-1 : i + 1])
    return key


def bigram_letter_dict(sentence_list):
    #function to obtain count of all letter bigrams in the train data
    #input is a list of strings
    #for each sentence in that list, we need to turn it in a list of words
    #then, each word in that list is traversed to obtain letter bigrams
    letter_bigrams_dict = {}
    for line in sentence_list:
        for one_word in line.split():
            for i in range(0, len(one_word)+1):
                key = make_letter_key(one_word, i)
                
                #once you have a bigram, check if it's already in the dictionary
                #if so, add 1 to the count
                #if not, start count from 1    
                if key in letter_bigrams_dict:
                    letter_bigrams_dict[key]+=1
                else:
                    letter_bigrams_dict[key] = 1
    return letter_bigrams_dict


def unigram_letter_dict(sentence_list):
    #function to obtain count of all letter unigrams in the train data
    #input is a list of strings
    #for each sentence in that list, we need to turn it in a list of words
    #then, each word in that list is traversed to obtain letter unigrams
    letter_unigrams_dict = {}
    num_words = 0
    for line in sentence_list:
        num_words += len(line.split())
        for word in line.split():
            for letter in word:
                if letter in letter_unigrams_dict:
                    letter_unigrams_dict[letter]+=1
                else:
                    letter_unigrams_dict[letter] = 1
    #total number of words gives us the number of times we saw the special chars
    #'beginning of word' and 'end of word'
    letter_unigrams_dict['<b>'] = num_words
    letter_unigrams_dict['<e>'] = num_words
    return letter_unigrams_dict


def score_letters(sentence, uniletter, biletter):
    #this function takes a sentence, a python dict with count of unigrams in train
    #and a python dict with count of bigrams in train and return the add-one smoothed
    #probability that the provided sentence is in the language of the two dictionaries
    #this is the function that performs the language model scoring for letter bigrams
    #note that smoothing is done on the fly for counts and smoothed counts are
    #then turned into probabilities on the fly

    #V is the size of the vocabulary, ie the count of distinct unigrams
    V = len(uniletter)
    score = 0
    #the input is a single sentence
    #turn that sentence into a list of words
    #then traverse each word to compute letter bigrams
    #unigram for conditional proba is just the first character in the bigram
    for one_word in sentence.split():
        for i in range(0, len(one_word)+1):
            bigram_key = make_letter_key(one_word, i)
            unigram_key = bigram_key.split(',')[0]
            
            #this is how we deal with unknowns
            #an unknown is a unigram/bigram that was never seen in training
            #an unknown is therefore NOT one of the keys in the python dict with the
            #counts of unigrams/bigrams in training
            #python raises a KeyError when asked to go in a dict and return value
            #associated to non-existing key; exploit this to deal with unknowns
            try:
                numerator = (biletter[bigram_key] + 1)
            except KeyError:
                numerator = 1
            
            try:
                denominator = uniletter[unigram_key] + V
            except KeyError:
                denominator = V
                
            proba = np.log(numerator/denominator)
            score += proba
    return score


def make_output_letter(test,
                       uniletter_dict_eng, biletter_dict_eng,
                       uniletter_dict_ita, biletter_dict_ita,
                       uniletter_dict_fra, biletter_dict_fra):
    #this function takes the test sentences and a series of unigram/bigram counts
    #and returns a list with the predicted language for each sentence in the test set
    #for each sentence, model is scored using the ENG, ITA and FRA dictionaries
    #then, the predicted language is simply the one associated to the highest probability
    output = []
    for i in range(len(test)):
        test_line = test[i]
        results = {'English': score_letters(test_line, uniletter_dict_eng, biletter_dict_eng),
                   'Italian': score_letters(test_line, uniletter_dict_ita, biletter_dict_ita),
                   'French' : score_letters(test_line, uniletter_dict_fra, biletter_dict_fra)}
        predicted_language = max(results.items(), key=operator.itemgetter(1))[0]
        output.append('{} {}'.format(i+1, predicted_language))
    return output


def main():
    #read train data
    ita = read_files('LangId.train.Italian')
    fra = read_files('LangId.train.French')
    eng = read_files('LangId.train.English')
    #read test data
    test = read_files('LangId.test')
    #read ground truth, ie the solution
    with open('LangId.sol', 'r') as file:
        sol = file.readlines()
        sol = [x.replace('\n', '') for x in sol]

    #create letter unigram counts
    uniletter_dict_ita = unigram_letter_dict(ita)
    uniletter_dict_fra = unigram_letter_dict(fra)
    uniletter_dict_eng = unigram_letter_dict(eng)

    #create letter bigram counts
    biletter_dict_ita = bigram_letter_dict(ita)
    biletter_dict_fra = bigram_letter_dict(fra)
    biletter_dict_eng = bigram_letter_dict(eng)

    #create list with predicted languages of all sentences in test data
    output_letter = make_output_letter(test,
                       uniletter_dict_eng, biletter_dict_eng,
                       uniletter_dict_ita, biletter_dict_ita,
                       uniletter_dict_fra, biletter_dict_fra)

    #write output to file
    write_out('letterLangId.out', output_letter)

    #write performance to the console
    perf = compute_performance(output_letter, sol)
    print('Accuracy of letter bigram model with add-one smoothing: {}%'.format(perf))


if __name__ == "__main__":
    main()
