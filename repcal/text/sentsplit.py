#!/usr/bin/env python
# -*- coding: utf-8 -*-

from nltk.tokenize import sent_tokenize
from typing import List

def get_sentsplitter(name, linebreak=True):
    if name == 'linebreak':
        return LineBreakSplitter()
    elif name == 'nltk':
        return NLTKSentenceSplitter(linebreak)
    elif name == 'none':
        return NullSentenceSplitter()
    else:
        raise ValueError('Unknown sentence splitter: %s' % name)


class LineBreakSplitter:
    @staticmethod
    def split(text) -> List[str]:
        return text.split('\n')


class NLTKSentenceSplitter:
    def __init__(self, linebreak=True):
        self.linebreak = linebreak

    def split(self, text) -> List[str]:
        sents = []
        for sent in sent_tokenize(text):
            if self.linebreak:
                for sent2 in sent.split('\n'):
                    sents.append(sent2)
            else:
                sents.append(sent)
        return sents


class NullSentenceSplitter:
    @staticmethod
    def split(text) -> List[str]:
        return [text]
