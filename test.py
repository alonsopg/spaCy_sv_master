import spacy
import random
from spacy.gold import GoldParse

TRAIN_DATA = [
    ('Vem är Shaka Khan?', {
        'entities': [(7, 17, 'PER')]
    }),
    ('Zhang Hen, ha en trevlig dag!', {
        'entities': [(0, 10, 'PER')]
    }),
    ('Jag älskar dig, Zhang Heng.', {
        'entities': [(16, 26, 'PER')]
    }),
    ('Jag gillar London och Berlin.', {
        'entities': [(11, 17, 'LOC'), (22, 28, 'LOC')]
    })
]

nlp = spacy.blank('sv').from_disk('sv_model')

optimizer = nlp.begin_training()
for i in range(20):
    random.shuffle(TRAIN_DATA)
    for text, annotations in TRAIN_DATA:
        nlp.update([text], [annotations], sgd=optimizer)
nlp.to_disk('sv_model')


test_text = 'Jag saknar dig, Zhang Heng.'
doc = nlp(test_text)
# print("Entities in '%s'" % test_text)
for ent in doc.ents:
    print(ent.label_, ent.text)