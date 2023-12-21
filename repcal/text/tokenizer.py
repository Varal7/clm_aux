from nltk.tokenize import wordpunct_tokenize

from typing import List

def get_tokenizer(name, port=None):
    if name == 'nltk':
        return NLTKTokenizer()
    elif name == 'whitespace':
        return WhiteSpaceTokenizer()
    else:
        raise ValueError('Unknown tokenizer: %s' % name)


class NLTKTokenizer:
    @staticmethod
    def tokenize(text) -> List[str]:
        return wordpunct_tokenize(text)

class WhiteSpaceTokenizer:
    @staticmethod
    def tokenize(text) -> List[str]:
        return text.split()
