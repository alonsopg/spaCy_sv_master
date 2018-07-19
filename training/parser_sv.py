# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import string
import re
from spacy.lang.sv import Swedish

# file_name = "./data/parser/sv_pud-ud-test.conllu"
file_name = "./data/parser/text.txt"
# file_name = "./data/parser/sv_talbanken-ud-train.conllu"

def input_fn(file_name):
    train_data = []
    with open(file_name, 'r', encoding='utf-8') as f:
        heads = []
        deps = []
        annotations = []
        texts = []
        for line in f.readlines():
            if re.match("^# text = ", line):
                texts.append(line.lstrip("# text = ").rstrip('\n'))
            elif re.match("^#", line):
                del line
            elif not line.strip() == "":
                sent = line.lstrip()
                lines = [line.split('\t') for line in sent.split('\n')][0]
                if lines[6] == "_":
                    del lines
                else:
                    heads.append(int(lines[6]))
                    deps.append(lines[7])
            elif line.strip() == "":
                annotations.append([heads, deps])
                heads = []
                deps = []

        for i in range(len(annotations)):
            # Encode per-token tags following the BILUO scheme into entity offsets.
            text = texts[i]
            heads, deps = annotations[i][0], annotations[i][1]
            nlp = Swedish()
            spacy_doc = nlp(text)
            train_format = (str(spacy_doc), {
                'heads': (heads),
                'deps': (deps)
            })
            train_data.append(train_format)
    return train_data


if __name__ == '__main__':
    train_data = input_fn(file_name)
    print(train_data)
