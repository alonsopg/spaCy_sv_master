#!/usr/bin/env python
# coding: utf8
"""Example of training spaCy dependency parser, starting off with an existing
model or a blank model. For more details, see the documentation:
* Training: https://spacy.io/usage/training
* Dependency Parse: https://spacy.io/usage/linguistic-features#dependency-parse

Compatible with: spaCy v2.0.0+
"""
from __future__ import unicode_literals, print_function

import plac
import random
from pathlib import Path
import spacy
import tqdm
from spacy.gold import GoldParse

from parser_sv import input_fn

from spacy.syntax.nonproj import preprocess_training_data

# training data
# TRAIN_DATA = [
#     ('Jag gillar London och Berlin.', {
#         'heads': [1, 1, 1, 2, 2, 1],
#         'deps': ['nsubj', 'ROOT', 'dobj', 'cc', 'conj', 'punct']
#     }),
#     ('Både för barnen och för deras föräldrar är det viktigt att föräldrarna får möjlighet att välja den livsform som de trivs bäst med.', {
#         'heads': [3, 3, 10, 7, 7, 7, 3, 10, 10, 0, 13, 13, 10, 13, 16, 14, 18, 16, 21, 21, 18, 21, 19, 10],
#         'deps': ['advmod', 'case', 'obl', 'cc', 'case', 'nmod:poss', 'conj', 'cop', 'expl', 'root', 'mark', 'nsubj',
#                  'csubj', 'obj', 'mark', 'acl', 'det', 'obj', 'obl', 'nsubj', 'acl:relcl', 'advmod', 'case', 'punct']
#     }),
#     ('Genom skattereformen införs individuell beskattning (särbeskattning) av arbetsinkomster.', {
#         'heads': [3, 1, 0, 5, 3, 5, 5, 5, 5, 9, 3],
#         'deps': ['case', 'obl', 'root', 'amod', 'nsubj:pass', 'punct', 'appos', 'punct', 'case', 'nmod', 'punct']
#     }),
#     ('Individuell beskattning av arbetsinkomster', {
#         'heads': [2, 0, 2, 3],
#         'deps': ['amod', 'root', 'case', 'nmod']
#     })
# ]

TRAIN_DATA = input_fn("./data/parser/text.txt")
# TRAIN_DATA = input_fn("./data/parser/talbanken-stanford-train.conll")
# TRAIN_DATA = input_fn("./data/UD_Swedish/sv-ud-train.conllu")
# TRAIN_DATA = preprocess_training_data(TRAIN_DATA)
# print(TRAIN_DATA)

@plac.annotations(
    model=("Model name. Defaults to blank 'en' model.", "option", "m", str),
    output_dir=("Optional output directory", "option", "o", Path),
    n_iter=("Number of training iterations", "option", "n", int))
def main(model=None, output_dir=None, n_iter=1):
    """Load the model, set up the pipeline and train the parser."""
    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank('en')  # create blank Language class
        print("Created blank 'en' model")

    # add the parser to the pipeline if it doesn't exist
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if 'parser' not in nlp.pipe_names:
        parser = nlp.create_pipe('parser')
        nlp.add_pipe(parser, first=True)
    # otherwise, get it, so we can add labels to it
    else:
        parser = nlp.get_pipe('parser')

    # add labels to the parser
    for _, annotations in TRAIN_DATA:
        for dep in annotations.get('deps', []):
            parser.add_label(dep)


    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'parser']
    with nlp.disable_pipes(*other_pipes):  # only train parser
        optimizer = nlp.begin_training(lambda: [])
        for itn in range(n_iter):
            random.shuffle(TRAIN_DATA)
            losses = {}
            for text, annotations in tqdm.tqdm(TRAIN_DATA):
                doc = nlp(text)
                # print(text)
                # test GoldParse
                annotations = GoldParse(doc, heads=annotations.get('heads'), deps=annotations.get('deps'), make_projective=True)
                nlp.update([text], [annotations], drop=0.5, sgd=optimizer, losses=losses)
            print("n_iter: {}, loss: {}".format(itn, losses['parser']))

    # test the trained model
    test_text = "Den gäller även för oskifta dödsbon och familjestiftelser."
    doc = nlp(test_text)
    print('Dependencies', [(t.text, t.dep_, t.head.text) for t in doc])

    # save model to output directory
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

        # test the saved model
        print("Loading from", output_dir)
        nlp2 = spacy.load(output_dir)
        doc = nlp2(test_text)
        print('Dependencies', [(t.text, t.dep_, t.head.text) for t in doc])

if __name__ == '__main__':
    plac.call(main)