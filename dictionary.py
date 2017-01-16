import json

paths = [ "./data/lilypond/TestSet/",
        "./data/lilypond/TrainSet/"]
dict_path = "./data/lilypond/dictionary.json"

#
# Create dictionary mapped as "music_symbol" : "number" and export to json file
#
def create_dictionary():
    categories = []
    print("Creating dictionary...")

    for path in paths:
        f = open(path + "labels.txt")
        for line in f:
            for token in line.split():
                if token not in categories:
                    categories.append(token)

    words = dict()
    for i,cat in enumerate(categories):
        words[cat] = i

    print("Dictionary has " + str(len(words)) + " words")
    with open(dict_path,'w') as fp:
        json.dump(words, fp, sort_keys=True, indent=4)

def translate():
    words = None
    with open(dict_path,'r') as fp:
        words = json.load(fp )
    print(words)

create_dictionary()
translate()
