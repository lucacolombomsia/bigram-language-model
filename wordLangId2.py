from support import read_files, text_preprocess, write_out, compute_performance
#note that some functions are identical across the two word-based models
#hence, we import those from the other script
from wordLangId import make_word_key, bigram_word_dict, unigram_word_dict
import numpy as np
import operator
from collections import Counter


def turing_smoothing_dict(uniword, biword):
    #this function returns a series of values that we will need to apply GT smoothing
    #first of all, it computes -for both unigrams and bigrams- the frequency of frequencies vector
    #this is a vector that, for each value x, shows how many tokens were observed x times
    #we need a vector that collects Nx's for all x's, where Nx is the number of N-grams that
    #occur x times in the training corpus
    #it will also com
    turing_bigram_dict = {}
    #need to compute the total number of possible bigrams
    #this will allow us to count number of unknown bigrams
    #total number of possible bigrams is the square of the number of unigrams
    #number of unknown bigrams is given by the following formula
    #"number of total possible bigrams minus number of known bigrams"
    total_number_bigrams = len(uniword)**2
    count_unknowns_bigrams = total_number_bigrams - len(biword)
    #need to get Nx for unigrams and bigrams
    Nx_unigram = dict(Counter(uniword.values()))
    Nx_bigram = dict(Counter(biword.values()))
    #need to account for bigrams we have never seen before
    Nx_bigram[0] = count_unknowns_bigrams
    #need to account for unigrams we have never seen before
    #only reasonable assumption here is that there is a 'unique' unknown token
    Nx_unigram[0] = 1
    #for GT to work, for each N(x) there must exist a N(x+1) different from zero
    #we will ensure that the property above holds by finding a cutoff that is 
    #the last Nx s.t. N(x+1) != 0
    #GT smoothing will be applied only to tokens that appeared x times or fewer
    #this is not a problem, as smoothing is more of a concern for tokens that were
    #observed few times than tokens that were observed many times
    #given that we are applying smoothing to some (but not all tokens) probabilities
    #will not sum to one, but this is again a minor concern
    #here we find the threshold for both bigrams and unigrams
    unigram_sorted_keys = sorted(Nx_unigram.keys())
    for i in range(0, len(unigram_sorted_keys)):
        if i != unigram_sorted_keys[i]:
            unigram_threshold = unigram_sorted_keys[i-1]
            break
    bigram_sorted_keys = sorted(Nx_bigram.keys())
    for i in range(0, len(bigram_sorted_keys)):
        if i != bigram_sorted_keys[i]:
            bigram_threshold = bigram_sorted_keys[i-1]
            break
    return Nx_unigram, unigram_threshold, Nx_bigram, bigram_threshold


def score_words_gt(sentence, uniword, biword, Nx_uni, uni_threshold, Nx_bi, bi_threshold):
    #this is the function that performs the language model scoring for word bigrams
    #it takes a series of inputs:
    # - a sentence
    # - a python dict with count of unigrams in train
    # - a python dict with count of bigrams in train
    # - list of Nx's for unigrams
    # - threshold above which we don't apply GT smoothing to unigram counts
    # - list of Nx's for bigrams
    # - threshold above which we don't apply GT smoothing to bigram counts
    #the function returns the GT smoothed probability that the provided sentence is in
    #the language of the 2 dictionaries, the 2 lists of Nx's and the 2 thresholds
    #note that counts are smoothed using GT and turned into probabilities on the fly
    #also note that GT smoothing will be applied only to tokens that appeared
    #less than a given amount of times (as per the threshold parameters of this function)

    score = 0
    #the input is a single sentence
    #turn that sentence into a list of words
    #then traverse list of words to find word bigrams
    #unigram for conditional proba is just the first word in the bigram
    words = sentence.split()
    for i in range(0, len(words)+1):
        bigram_key = make_word_key(words, i)
        unigram_key = bigram_key.split(' ')[0]
        
        #start by finding the count associated to the bigram and unigram
        #this is the true count, not the smoothed count!
        #hence, unknowns are given a 0 count!
        #note how we deal with unknowns
        #an unknown is a unigram/bigram that was never seen in training
        #an unknown is therefore NOT a key in the python dict with the
        #counts of unigrams/bigrams in training
        #python raises a KeyError when asked to go in a dict and return value
        #associated to non-existing key; exploit this to deal with unknowns
        try:
            bigram_real_count = biword[bigram_key]
        except KeyError:
            bigram_real_count = 0
            
        try:
            unigram_real_count = uniword[unigram_key]
        except KeyError:
            unigram_real_count = 0
            
        #now apply Good-Turing smoothing and turn counts into probabilities
        #as explained above, smoothing will only apply if the true count is below
        #the threshold computed in the function above and passed to this function
        #as a parameter
        if bigram_real_count < bi_threshold:
            bigram_gt_proba = ((bigram_real_count + 1) * 
                               Nx_bi[bigram_real_count + 1] / 
                               (Nx_bi[bigram_real_count]*sum(biword.values())))
        else:
            bigram_gt_proba = bigram_real_count/sum(biword.values())
        
        if unigram_real_count < uni_threshold:
            unigram_gt_proba = ((unigram_real_count + 1) *
                               Nx_uni[unigram_real_count + 1] /
                               (Nx_uni[unigram_real_count]*sum(uniword.values())))
        else:
            unigram_gt_proba = unigram_real_count/sum(uniword.values())
        
        #now that we have probabilities, compute conditional probability    
        cond_proba = np.log(bigram_gt_proba/unigram_gt_proba)
        score += cond_proba
    return score


def make_output_words_gt(test,
                       uniword_dict_eng, biword_dict_eng,
                         Nx_unigram_eng, unigram_threshold_eng, Nx_bigram_eng, bigram_threshold_eng,
                       uniword_dict_ita, biword_dict_ita,
                         Nx_unigram_ita, unigram_threshold_ita, Nx_bigram_ita, bigram_threshold_ita,
                       uniword_dict_fra, biword_dict_fra,
                         Nx_unigram_fra, unigram_threshold_fra, Nx_bigram_fra, bigram_threshold_fra):
    #this function takes the test sentences and returns a list with the predicted
    #language for each sentence in the test set
    #for each sentence, model is scored using the ENG, ITA and FRA dictionaries
    #the predicted language is the language that is associated to the highest proba
    #since we are calling the score_words_gt function, we need to pass a fair
    #bit of parameters to this function
    #for each language, we need the count of unigrams and bigrams, the lists of Nx's
    #and the threshold for GT smoothing
    output = []
    for i in range(len(test)):
        test_line = test[i]
        results = {'English': score_words_gt(test_line, uniword_dict_eng, biword_dict_eng,
                                            Nx_unigram_eng, unigram_threshold_eng, Nx_bigram_eng, bigram_threshold_eng),
                   'Italian': score_words_gt(test_line, uniword_dict_ita, biword_dict_ita,
                                            Nx_unigram_ita, unigram_threshold_ita, Nx_bigram_ita, bigram_threshold_ita),
                   'French' : score_words_gt(test_line, uniword_dict_fra, biword_dict_fra,
                                            Nx_unigram_fra, unigram_threshold_fra, Nx_bigram_fra, bigram_threshold_fra)}
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

    #compute Nx's lists and thresholds for unigrams and bigrams
    Nx_unigram_ita, unigram_threshold_ita, Nx_bigram_ita, bigram_threshold_ita = turing_smoothing_dict(uniword_dict_ita, biword_dict_ita)
    Nx_unigram_fra, unigram_threshold_fra, Nx_bigram_fra, bigram_threshold_fra = turing_smoothing_dict(uniword_dict_fra, biword_dict_fra)
    Nx_unigram_eng, unigram_threshold_eng, Nx_bigram_eng, bigram_threshold_eng = turing_smoothing_dict(uniword_dict_eng, biword_dict_eng)

    #create list with predicted languages of all sentences in test data
    output_word_gt = make_output_words_gt(test,
                       uniword_dict_eng, biword_dict_eng,
                         Nx_unigram_eng, unigram_threshold_eng, Nx_bigram_eng, bigram_threshold_eng,
                       uniword_dict_ita, biword_dict_ita,
                         Nx_unigram_ita, unigram_threshold_ita, Nx_bigram_ita, bigram_threshold_ita,
                       uniword_dict_fra, biword_dict_fra,
                         Nx_unigram_fra, unigram_threshold_fra, Nx_bigram_fra, bigram_threshold_fra)

    #write output to file
    write_out('wordLangId2.out', output_word_gt)

    #write performance to the console
    perf = compute_performance(output_word_gt, sol)
    print('Accuracy of word bigram model with Good-Turing smoothing: {}%'.format(perf))


if __name__ == "__main__":
    main()
