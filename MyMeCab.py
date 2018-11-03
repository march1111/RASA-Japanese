# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# from __future__ import unicode_literals

import MeCab

from typing import Any, List, Text

from rasa_nlu.components import Component
from rasa_nlu.config import RasaNLUModelConfig
from rasa_nlu.tokenizers import Tokenizer, Token
from rasa_nlu.training_data import Message
from rasa_nlu.training_data import TrainingData




class Mecab(Tokenizer, Component):
    name = "Mecab"

    provides = ["tokens"]

    INDEX_CATEGORY = 0
    INDEX_ROOT_FORM = 6
    TARGET_CATEGORIES = ["名詞", "動詞",  "形容詞", "副詞", "連体詞", "感動詞"]

    def train(self, training_data, config, **kwargs):
        # type: (TrainingData, RasaNLUModelConfig, **Any) -> None

        for example in training_data.training_examples:
            example.set("tokens", self.tokenize(example.text))

    def process(self, message, **kwargs):
        # type: (Message, **Any) -> None

        message.set("tokens", self.tokenize(message.text))


    def tokenize(self, text):
        #type: (Text) -> List[Token]

        self.dictionary = "mecabrc"
        self.tagger = MeCab.Tagger(self.dictionary)

        if not text:
            return []

        words = []
        if type(text) != str:
            text = u''.join((text)).encode('utf-8')
        node = self.tagger.parseToNode(str(text))

        running_offset = 0
        word_offset = 0

        while node:
            features = node.feature.split(',')
            if features[self.INDEX_CATEGORY] in self.TARGET_CATEGORIES:
                if features[self.INDEX_ROOT_FORM] == "*":
                    word_offset = text.index(node.surface, running_offset)
                    word_len = len(node.surface)
                    running_offset = word_offset + word_len
                    words.append(Token(node.surface, word_offset))
                else:
                    try:
                        word_offset = text.index(features[self.INDEX_ROOT_FORM], running_offset)
                        word_len = len(features[self.INDEX_ROOT_FORM])
                        running_offset = word_offset + word_len
                        words.append(Token(features[self.INDEX_ROOT_FORM], word_offset))
                    except ValueError:
                        print("No such a string")
                        if not word_offset:
                            word_offset = 0
                        word_len = 1
                        running_offset = word_offset + word_len

            node = node.next
        # for eachword in words:
        #     print('Word ==> {}'.format(eachword))
        return words
