import json
import sys

path = sys.argv[1]
dict_path = path + "dictionary.json"

#
# Create dictionary mapped as "music_symbol" : "number" and export to json file
#
def create_dictionary():
    categories = []
    max_length = 0
    print("Creating dictionary...")

    f = open(path + "labels.txt")
    for line in f:
        for (i,token) in enumerate(line.split()):
            if token not in categories:
                categories.append(token)
            if i > max_length:
                max_length = i

    words = dict()
    for i,cat in enumerate(categories):
        words[cat] = i

    dict_length = len(words)
    print("Dictionary has " + str(dict_length) + " words")
    max_length += 1
    print("Maximum length is " + str(max_length))
    with open(dict_path,'w') as fp:
        json.dump(words, fp, sort_keys=True, indent=4)

    print("---> Succesfully imported to JSON format")
    return max_length, dict_length

#
# Translate the labels to a coded format using the dictionary
#
def codify(max_length, dict_length):
    words = None
    print("Codifying labels...")
    with open(dict_path,'r') as fp:
        words = json.load(fp)

    f = open(path + "labels.txt")
    aux = ""
    for line in f:
        tokens = line.split()
        for i in range(len(tokens)):
            aux = aux + str(words[tokens[i]]) + ' '
        aux = aux + '\n'

    f = open(path + "labels_cod.txt", 'w')
    f.write(aux)
    print("---> Succesfully codified labels")


max_length, dict_length = create_dictionary()
codify(max_length, dict_length)
