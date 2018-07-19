# coding: utf8
from __future__ import unicode_literals

"""
# origion import

from .tokenizer_exceptions import TOKENIZER_EXCEPTIONS
from .stop_words import STOP_WORDS
from .morph_rules import MORPH_RULES
from .lemmatizer import LEMMA_RULES, LOOKUP

from ..tokenizer_exceptions import BASE_EXCEPTIONS
from ..norm_exceptions import BASE_NORMS
from ...language import Language
from ...attrs import LANG, NORM
from ...util import update_exc, add_lookups
"""

from .tokenizer_exceptions import TOKENIZER_EXCEPTIONS
from .norm_exceptions import NORM_EXCEPTIONS
from .tag_map import TAG_MAP
from .stop_words import STOP_WORDS
from .lex_attrs import LEX_ATTRS
from .morph_rules import MORPH_RULES
from .lemmatizer import LEMMA_RULES, LEMMA_INDEX, LEMMA_EXC, LOOKUP
from .syntax_iterators import SYNTAX_ITERATORS

from ..tokenizer_exceptions import BASE_EXCEPTIONS
from ..norm_exceptions import BASE_NORMS
from ...language import Language
from ...attrs import LANG, NORM
from ...util import update_exc, add_lookups


class SwedishDefaults(Language.Defaults):
    lex_attr_getters = dict(Language.Defaults.lex_attr_getters)
    # update lex_attr_getters
    lex_attr_getters.update(LEX_ATTRS)
    lex_attr_getters[LANG] = lambda text: 'sv'
    lex_attr_getters[NORM] = add_lookups(Language.Defaults.lex_attr_getters[NORM],
                                         BASE_NORMS, NORM_EXCEPTIONS)
    tokenizer_exceptions = update_exc(BASE_EXCEPTIONS, TOKENIZER_EXCEPTIONS)
    # tag_map
    tag_map = TAG_MAP
    stop_words = STOP_WORDS
    # morph_rules
    morph_rules = MORPH_RULES

    lemma_rules = LEMMA_RULES
    lemma_lookup = LOOKUP

    # lemma_index, lemma_exc
    lemma_index = LEMMA_INDEX
    lemma_exc = LEMMA_EXC
    
    # syntax_iterators
    syntax_iterators = SYNTAX_ITERATORS


class Swedish(Language):
    lang = 'sv'
    Defaults = SwedishDefaults


__all__ = ['Swedish']
