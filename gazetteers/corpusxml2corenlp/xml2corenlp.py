from bs4 import BeautifulSoup, Tag
import os, sys
import fileinput
from tqdm import tqdm

f1 = open(sys.argv[2], "a")
sentence = []
train_sentence = []

doc = []
with open(sys.argv[1], "r") as f:
    for line in tqdm(f):
        soup = BeautifulSoup(line, "html.parser")
        for word in soup.find_all('w'):
                if word['pos'] == "PM":
                    train_sentence.append(word.getText() + "\t"+ "LABEL")
                else:
                    train_sentence.append(word.getText() +"\t"+"0")

        if "</sentence>" in line:
            doc.append("\n".join(train_sentence) + "\n")
            train_sentence = []

        if len(doc) == 500:
            f1.write("".join(doc))
            doc = []

f1.write("".join(doc))
f1.close()
