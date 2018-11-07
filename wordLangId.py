from support import read_files, text_preprocess, write_out, compute_performance
import numpy as np
import operator


def make_word_key(words, i):
    #this function should be called while iterating through the words of a sentence
    #it return the words bigram
    #special characters <b> and <e> are used to identify beginning and
    #end of sentence respectively
    #two words in bigram are separated by a whitespace
    if i == 0:
        key = '<b> ' + words[i]
    elif i == len(words):
        key = words[i-1] + ' <e>'
    else:
        key = ' '.join(words[i-1 : i + 1])
    return key


def bigram_word_dict(sentence_list):
    #function to obtain count of each word bigram in the train corpus
    #input is a list of strings
    #for each sentence in that list, we need to turn it in a list of words
    #then, we traverse the list of words and create word bigrams
    word_bigrams_dict = {}
    for line in sentence_list:
        words = line.split()
        for i in range(0, len(words)+1):
            key = make_word_key(words, i)
            
            #once you have a bigram, check if it's already in the dictionary
            #if so, add 1 to the count
            #if not, start count from 1 
            if key in word_bigrams_dict:
                word_bigrams_dict[key]+=1
            else:
                word_bigrams_dict[key] = 1
    return word_bigrams_dict


def unigram_word_dict(sentence_list):
    #function to obtain count of each word unigram in the train corpus
    #input is a list of strings
    #for each sentence in that list, we need to turn it in a list of words
    #then, we traverse the list of words and create word unigrams
    word_unigrams_dict = {}
    for line in sentence_list:
        for word in line.split():
            if word in word_unigrams_dict:
                word_unigrams_dict[word]+=1
            else:
                word_unigrams_dict[word] = 1
    #total number of sentences gives us the number of times we saw the special chars
    #'beginning of sentence' and 'end of sentence'
    word_unigrams_dict['<b>'] = len(sentence_list)
    word_unigrams_dict['<e>'] = len(sentence_list)
    return word_unigrams_dict


def score_words(sentence, uniword, biword):
    #this is the function that performs the language model scoring for word bigrams
    #it takes a sentence, a python dict with count of unigrams in train
    #and a python dict with count of bigrams in train and returns the add-one smoothed
    #probability that the provided sentence is in the language of the two dictionaries
    #note that smoothing is done on the fly for counts and smoothed counts are
    #then turned into probabilities on the fly using MLE

    #V is the size of the vocabulary, ie the count of distinct unigrams
    V = len(uniword)
    score = 0
    #the input is a single sentence
    #turn that sentence into a list of words
    #then traverse list of words to find word bigrams
    #unigram for conditional proba is just the first word in the bigram
    words = sentence.split()
    for i in range(0, len(words)+1):
        bigram_key = make_word_key(words, i)
        unigram_key = bigram_key.split(' ')[0]
        
        #here we perform the add-one smoothing of the counts
        #note how we deal with unknowns
        #an unknown is a unigram/bigram that was never seen in training
        #an unknown is therefore NOT a key in the python dict with the
        #counts of unigrams/bigrams in training
        #python raises a KeyError when asked to go in a dict and return value
        #associated to non-existing key; exploit this to deal with unknowns
        try:
            numerator = (biword[bigram_key] + 1)
        except KeyError:
            numerator = 1
        
        try:
            denominator = uniword[unigram_key] + V
        except KeyError:
            denominator = V
        
        #turn counts into conditional probability using MLE 
        proba = np.log(numerator/denominator)
        score += proba
    return score


def make_output_words(test,
                       uniword_dict_eng, biword_dict_eng,
                       uniword_dict_ita, biword_dict_ita,
                       uniword_dict_fra, biword_dict_fra):
    #this function takes the test sentences and a series of unigram/bigram counts
    #and returns a list with the predicted language for each sentence in the test set
    #for each sentence, model is scored using the ENG, ITA and FRA dictionaries
    #the predicted language is the language that is associated to the highest proba
    output = []
    for i in range(len(test)):
        test_line = test[i]
        results = {'English': score_words(test_line, uniword_dict_eng, biword_dict_eng),
                   'Italian': score_words(test_line, uniword_dict_ita, biword_dict_ita),
                   'French' : score_words(test_line, uniword_dict_fra, biword_dict_fra)}
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

    #create word unigram counts
    uniword_dict_ita = unigram_word_dict(ita)
    uniword_dict_fra = unigram_word_dict(fra)
    uniword_dict_eng = unigram_word_dict(eng)

    #create word bigram counts
    biword_dict_ita = bigram_word_dict(ita)
    biword_dict_fra = bigram_word_dict(fra)
    biword_dict_eng = bigram_word_dict(eng)

    #create list with predicted languages of all sentences in test data
    output_word = make_output_words(test,
                       uniword_dict_eng, biword_dict_eng,
                       uniword_dict_ita, biword_dict_ita,
                       uniword_dict_fra, biword_dict_fra)

    #write output to file
    write_out('wordLangId.out', output_word)

    #write performance to the console
    perf = compute_performance(output_word, sol)
    print('Accuracy of word bigram model with add-one smoothing: {}%'.format(perf))

if __name__ == "__main__":
    main()
