# coding: utf8
from __future__ import unicode_literals

from ...attrs import LIKE_NUM


_num_words = ['noll', 'ett', 'två', 'tre', 'fyra', 'fem', 'sex', 'sju',
              'åtta', 'nio', 'tio', 'elva', 'tolv', 'tretton', 'fjorton',
              'femton', 'sexton', 'sjutton', 'arton', 'nitton', 'tjugo',
              'trettio', 'fyrtio', 'femtio', 'sextio', 'sjuttio', 'åttio', 'nittio',
              'hundra', 'ettusen', 'miljon', 'miljarder', 'biljon', 'kvadriljon',
              'gajillion', 'bazillion']


def like_num(text):
    text = text.replace(',', '').replace('.', '')
    if text.isdigit():
        return True
    if text.count('/') == 1:
        num, denom = text.split('/')
        if num.isdigit() and denom.isdigit():
            return True
    if text.lower() in _num_words:
        return True
    return False


LEX_ATTRS = {
    LIKE_NUM: like_num
}
