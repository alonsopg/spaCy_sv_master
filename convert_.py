import spacy
from spacy.cli.convert import convert

input_file = "data/UD_Swedish/sv-ud-test.conllu"
output_dir = "data/ud_json"


"""
CONVERTERS = {
    'conllu': conllu2json,
    'conll': conllu2json,
    'ner': conll_ner2json,
    'iob': iob2json,
}

@plac.annotations(
    input_file=("input file", "positional", None, str),
    output_dir=("output directory for converted file", "positional", None, str),
    n_sents=("Number of sentences per doc", "option", "n", int),
    converter=("Name of converter (auto, iob, conllu or ner)", "option", "c", str),
    morphology=("Enable appending morphology to tags", "flag", "m", bool))
"""

convert(input_file, output_dir, n_sents=1, converter='conllu', morphology=True)