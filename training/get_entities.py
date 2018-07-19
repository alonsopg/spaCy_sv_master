# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import string
from spacy.gold import offsets_from_biluo_tags
from spacy.lang.sv import Swedish

# file_name = "text.txt"
file_name = "./data/ner/test_corpus.txt"

def input_fn(file_name):

    with open(file_name, 'r', encoding='utf-8') as f:
        docs = []
        tags = []
        instances = []
        for line in f.readlines():
            if not line.strip() == "":
                text = line.split('\t')
                docs.append(text[0])
                tags.append(text[-1].split('\n')[0])
            if line.strip() == "":
                instances.append([docs, tags])
                docs = []
                tags = []
    return instances

def convert_biluo(instances):

    train_data = []
    num_instances = len(instances)  # get the num of instances
    # get docs and tags in every instance
    for i in range(num_instances):
        docs = instances[i][0]
        tags = instances[i][1]

        # convert tags into biluo format
        num_tags = len(tags)    # tag num of every instance
        org_tags = []
        for org_index in range(num_tags):
            org_tags.append(tags[org_index])
        for tag_index in range(num_tags):
            if tags[tag_index] is '0':
                tags[tag_index] = 'O'
                org_tags[tag_index] = 'O'
                continue
            elif tag_index == 0:
                if tags[tag_index] != org_tags[tag_index+1]:
                    tags[tag_index] = 'U-'+tags[tag_index]
                else:
                    tags[tag_index] = 'B-'+tags[tag_index]
            elif tag_index == num_tags-1:
                if tags[tag_index] != org_tags[tag_index-1]:
                    tags[tag_index] = 'U-'+tags[tag_index]
                else:
                    tags[tag_index] = 'L-'+tags[tag_index]
            else:
                if (tags[tag_index] != org_tags[tag_index-1]) and (tags[tag_index] != org_tags[tag_index+1]):
                    tags[tag_index] = 'U-'+tags[tag_index]
                elif (tags[tag_index] == org_tags[tag_index-1]) and (tags[tag_index] == org_tags[tag_index+1]):
                    tags[tag_index] = 'I-'+tags[tag_index]
                elif (tags[tag_index] == org_tags[tag_index-1]) and (tags[tag_index] != org_tags[tag_index+1]):
                    tags[tag_index] = 'L-'+tags[tag_index]
                elif (tags[tag_index] != org_tags[tag_index-1]) and (tags[tag_index] == org_tags[tag_index+1]):
                    tags[tag_index] = "B-"+tags[tag_index]

        texts = " ".join(docs).strip()

        # Encode per-token tags following the BILUO scheme into entity offsets.
        nlp = Swedish()
        spacy_doc = nlp(texts)
        entities = offsets_from_biluo_tags(spacy_doc, tags)
        train_format = (texts, {
            'entities': (entities)
        })
        train_data.append(train_format)

    return train_data

if __name__ == '__main__':
    instances = input_fn(file_name)

    tain_data = convert_biluo(instances)
    for i in range(2):
        print(tain_data[i])

