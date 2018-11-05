import re
import string
import operator

def read_files(path):
    with open(path, 'rb',) as file:
        data = file.read().decode('utf8', 'surrogateescape')
        data = data.splitlines()
        #want to remove punctuation, make lowercase and remove double spaces
        data = text_preprocess(data)
    #make sure we drop empty lines
    data = list(filter(None, data))
    return data


def text_preprocess(text_list):
    output = []
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
    N = len(output_list)
    count = 0
    for i in range(N):
        if output_list[i] == ground_truth[i]:
            count += 1
    return count
