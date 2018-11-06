import re
import string
import operator

def read_files(path):
    #function to read the training files and test file
    #the chosen encoding has proven to be optimal given the input data
    with open(path, 'rb',) as file:
        data = file.read().decode('utf8', 'surrogateescape')
        data = data.splitlines()
        #text_preprocess() removes punctuation, makes text all lowercase and
        #removes double/leading/trailing spaces
        data = text_preprocess(data)
    #make sure we drop empty lines
    data = list(filter(None, data))
    return data


def text_preprocess(text_list):
    #input is a list of strings
    #output will be a list of clean strings
    output = []
    #compile a regex with all punctuation marks
    regex = re.compile('[%s]' % re.escape(string.punctuation))
    for one_string in text_list:
        #remove punctuation
        one_string = regex.sub('', one_string)
        #make lowercase
        one_string = one_string.lower()
        #remove double spaces
        one_string = ' '.join(one_string.split())
        output.append(one_string)
    return output


def write_out(output_file, output_list):
    #write the final output in the output file
    with open(output_file, 'w') as file:
        for x in output_list:
            file.write(x + '\n')


def compute_performance(output_list, ground_truth):
    #function to check percentage accuracy of a specific language model
    #compares output list from a language model with ground truth
    N = len(output_list)
    #count variable will store number of correctly classified sentences
    count = 0
    for i in range(N):
        if output_list[i] == ground_truth[i]:
            count += 1
    return round(count/N*100, 1)
