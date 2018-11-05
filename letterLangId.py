from support import read_files, text_preprocess, write_out, compute_performance
import numpy as np
import operator

def make_letter_key(one_word, i):
    if i == 0:
        key = '<b>,' + one_word[i]
    elif i == len(one_word):
        key = one_word[i-1] + ',<e>'
    else:
        key = ','.join(one_word[i-1 : i + 1])
    return key


def bigram_letter_dict(sentence_list):
    letter_bigrams_dict = {}
    for line in sentence_list:
        for one_word in line.split():
            for i in range(0, len(one_word)+1):
                key = make_letter_key(one_word, i)
                    
                if key in letter_bigrams_dict:
                    letter_bigrams_dict[key]+=1
                else:
                    letter_bigrams_dict[key] = 1
    return letter_bigrams_dict


def unigram_letter_dict(sentence_list):
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
    letter_unigrams_dict['<b>'] = num_words
    letter_unigrams_dict['<e>'] = num_words
    return letter_unigrams_dict


def score_letters(sentence, uniletter, biletter):
    V = len(uniletter)
    score = 0
    for one_word in sentence.split():
        for i in range(0, len(one_word)+1):
            bigram_key = make_letter_key(one_word, i)
            unigram_key = bigram_key.split(',')[0]
            
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
    ita = read_files('LangId.train.Italian')
    fra = read_files('LangId.train.French')
    eng = read_files('LangId.train.English')
    test = read_files('LangId.test')
    with open('LangId.sol', 'r') as file:
        sol = file.readlines()
        sol = [x.replace('\n', '') for x in sol]

    uniletter_dict_ita = unigram_letter_dict(ita)
    uniletter_dict_fra = unigram_letter_dict(fra)
    uniletter_dict_eng = unigram_letter_dict(eng)

    biletter_dict_ita = bigram_letter_dict(ita)
    biletter_dict_fra = bigram_letter_dict(fra)
    biletter_dict_eng = bigram_letter_dict(eng)

    output_letter = make_output_letter(test,
                       uniletter_dict_eng, biletter_dict_eng,
                       uniletter_dict_ita, biletter_dict_ita,
                       uniletter_dict_fra, biletter_dict_fra)

    write_out('letterLangId.out', output_letter)

if __name__ == "__main__":
    main()
