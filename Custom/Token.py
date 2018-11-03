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


class WordDividor():
    INDEX_CATEGORY = 0
    INDEX_ROOT_FORM = 6
    TARGET_CATEGORIES = ["名詞", "動詞",  "形容詞", "副詞", "連体詞", "感動詞"]

    def __init__(self, dictionary="mecabrc"):
        self.dictionary = dictionary
        self.tagger = MeCab.Tagger(self.dictionary)

    def extract_words(self, text):
        self.TARGET_CATEGORIES = ["名詞", "動詞",  "形容詞", "副詞", "連体詞", "感動詞"]
        if not text:
            return []

        words = []
        if type(text) != str:
            text = u''.join((text)).encode('utf-8')
        node = self.tagger.parseToNode(str(text))
        while node:
            features = node.feature.split(',')
            #print features
            if features[self.INDEX_CATEGORY] in self.TARGET_CATEGORIES:
                if features[self.INDEX_ROOT_FORM] == "*":
                    words.append(node.surface)
                else:
                    words.append(features[self.INDEX_ROOT_FORM])

            node = node.next
        for eachword in words:
            print('Word ==> {}'.format(eachword))
        return words



class My_Mecab(Tokenizer, Component):
    name = "My_Mecab"

    provides = ["tokens"]

    # INDEX_CATEGORY = 0
    # INDEX_ROOT_FORM = 6
    # TARGET_CATEGORIES = ["名詞", "動詞",  "形容詞", "副詞", "連体詞", "感動詞"]

    # dictionary = "mecabrc"


    # def __init__(self, dictionary="mecabrc"):


    def train(self, training_data, config, **kwargs):
        # type: (TrainingData, RasaNLUModelConfig, **Any) -> None

        for example in training_data.training_examples:
            example.set("tokens", self.tokenize(example.text))

    def process(self, message, **kwargs):
        # type: (Message, **Any) -> None

        message.set("tokens", self.tokenize(message.text))


    def tokenize(self, text):
        indivi = WordDividor()
        data = indivi.extract_words(text)
        print(data)
        return data


        # self.dictionary = "mecabrc"
        # self.tagger = MeCab.Tagger(self.dictionary)
        #
        # if not text:
        #     return []
        #
        # words = []
        # if type(text) != str:
        #     text = u''.join((text)).encode('utf-8')
        # node = self.tagger.parseToNode(str(text))
        # while node:
        #     features = node.feature.split(',')
        #     #print features
        #     if features[self.INDEX_CATEGORY] in self.TARGET_CATEGORIES:
        #         if features[self.INDEX_ROOT_FORM] == "*":
        #             words.append(node.surface)
        #         else:
        #             words.append(features[self.INDEX_ROOT_FORM])
        #
        #     node = node.next
        # # for eachword in words:
        # #     print('Word ==> {}'.format(eachword))
        # return words



call = WordDividor()
print (call.extract_words("This is Me モバイルドメインリストの最新版が欲しい"))