from support import read_files, text_preprocess, write_out, compute_performance
import numpy as np
import operator
from collections import Counter


def make_word_key(words, i):
    if i == 0:
        key = '<b> ' + words[i]
    elif i == len(words):
        key = words[i-1] + ' <e>'
    else:
        key = ' '.join(words[i-1 : i + 1])
    return key


def bigram_word_dict(sentence_list):
    word_bigrams_dict = {}
    for line in sentence_list:
        words = line.split()
        for i in range(0, len(words)+1):
            key = make_word_key(words, i)
            
            if key in word_bigrams_dict:
                word_bigrams_dict[key]+=1
            else:
                word_bigrams_dict[key] = 1
    return word_bigrams_dict


def unigram_word_dict(sentence_list):
    word_unigrams_dict = {}
    for line in sentence_list:
        for word in line.split():
            if word in word_unigrams_dict:
                word_unigrams_dict[word]+=1
            else:
                word_unigrams_dict[word] = 1
    word_unigrams_dict['<b>'] = len(sentence_list)
    word_unigrams_dict['<e>'] = len(sentence_list)
    return word_unigrams_dict


def turing_smoothing_dict(uniword, biword):
    turing_bigram_dict = {}
    total_number_bigrams = len(uniword)**2
    count_unknowns_bigrams = total_number_bigrams - len(biword)
    #need to get Nx for unigrams and bigrams
    #these are the number of N-grams that occur xx times in the training text
    Nx_unigram = dict(Counter(uniword.values()))
    Nx_bigram = dict(Counter(biword.values()))
    #need to account for bigrams we have never seen before
    Nx_bigram[0] = count_unknowns_bigrams
    #need to account for unigrams we have never seen before
    Nx_unigram[0] = 1
    #for GT to work, for each N(x) there must exist a N(x+1) different from zero
    #we need to remove some values from the list of Nx's to ensure the property above holds
    #hence, we find a cutoff at the last Nx s.t. N(x+1) != 0
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
    score = 0
    words = sentence.split()
    for i in range(0, len(words)+1):
        bigram_key = make_word_key(words, i)
        unigram_key = bigram_key.split(' ')[0]
        
        ####start by finding the count! 0 is missing
        try:
            bigram_real_count = biword[bigram_key]
        except KeyError:
            bigram_real_count = 0
            
        try:
            unigram_real_count = uniword[unigram_key]
        except KeyError:
            unigram_real_count = 0
            
        ####then apply turing smoothing; smoothing will only apply if below threshold!!!
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
    ita = read_files('LangId.train.Italian')
    fra = read_files('LangId.train.French')
    eng = read_files('LangId.train.English')
    test = read_files('LangId.test')
    with open('LangId.sol', 'r') as file:
        sol = file.readlines()
        sol = [x.replace('\n', '') for x in sol]

    uniword_dict_ita = unigram_word_dict(ita)
    uniword_dict_fra = unigram_word_dict(fra)
    uniword_dict_eng = unigram_word_dict(eng)

    biword_dict_ita = bigram_word_dict(ita)
    biword_dict_fra = bigram_word_dict(fra)
    biword_dict_eng = bigram_word_dict(eng)

    Nx_unigram_ita, unigram_threshold_ita, Nx_bigram_ita, bigram_threshold_ita = turing_smoothing_dict(uniword_dict_ita, biword_dict_ita)
    Nx_unigram_fra, unigram_threshold_fra, Nx_bigram_fra, bigram_threshold_fra = turing_smoothing_dict(uniword_dict_fra, biword_dict_fra)
    Nx_unigram_eng, unigram_threshold_eng, Nx_bigram_eng, bigram_threshold_eng = turing_smoothing_dict(uniword_dict_eng, biword_dict_eng)

    output_word_gt = make_output_words_gt(test,
                       uniword_dict_eng, biword_dict_eng,
                         Nx_unigram_eng, unigram_threshold_eng, Nx_bigram_eng, bigram_threshold_eng,
                       uniword_dict_ita, biword_dict_ita,
                         Nx_unigram_ita, unigram_threshold_ita, Nx_bigram_ita, bigram_threshold_ita,
                       uniword_dict_fra, biword_dict_fra,
                         Nx_unigram_fra, unigram_threshold_fra, Nx_bigram_fra, bigram_threshold_fra)

    write_out('wordLangId2.out', output_word_gt)


if __name__ == "__main__":
    main()
