from support import read_files, text_preprocess, write_out, compute_performance
import numpy as np
import operator


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


def score_words(sentence, uniletter, biletter):
    V = len(uniletter)
    score = 0
    words = sentence.split()
    for i in range(0, len(words)+1):
        bigram_key = make_word_key(words, i)
        unigram_key = bigram_key.split(' ')[0]
        
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


def make_output_words(test,
                       uniword_dict_eng, biword_dict_eng,
                       uniword_dict_ita, biword_dict_ita,
                       uniword_dict_fra, biword_dict_fra):
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

    output_word = make_output_words(test,
                       uniword_dict_eng, biword_dict_eng,
                       uniword_dict_ita, biword_dict_ita,
                       uniword_dict_fra, biword_dict_fra)

    write_out('wordLangId.out', output_word)

if __name__ == "__main__":
    main()
