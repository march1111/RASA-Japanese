# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# from __future__ import unicode_literals

import re
from typing import Any, List, Text

from rasa_nlu.components import Component
from rasa_nlu.config import RasaNLUModelConfig
from rasa_nlu.tokenizers import Tokenizer, Token
from rasa_nlu.training_data import Message
from rasa_nlu.training_data import TrainingData


class Whitespace(Tokenizer, Component):
    name = "tokenizer_whitespace"

    provides = ["tokens"]

    def train(self, training_data, config, **kwargs):
        # type: (TrainingData, RasaNLUModelConfig, **Any) -> None

        for example in training_data.training_examples:
            example.set("tokens", self.tokenize(example.text))

    def process(self, message, **kwargs):
        # type: (Message, **Any) -> None

        message.set("tokens", self.tokenize(message.text))

    def tokenize(self, text):
        # type: (Text) -> List[Token]

        # there is space or end of string after punctuation
        # because we do not want to replace 10.000 with 10 000
        words = re.sub(r'[.,!?]+(\s|$)', ' ', text).split()

        running_offset = 0
        tokens = []
        for word in words:
            print("Word: {}".format(word))
            word_offset = text.index(word, running_offset)
            word_len = len(word)
            running_offset = word_offset + word_len

            print("Ofset {}".format(word_offset))
            print("Word Length: {}".format(word_len))
            print("Running ofSet: {}".format(running_offset))

            tokens.append(Token(word, word_offset))
        #
        # print("Ofset {}".format(word_offset))
        # print("Word Length: {}".format(word_len))
        # print("Running ofSet: {}".format(running_offset))


        print("========================================")
        print(tokens)
        print(dir(tokens))

        return tokens




print("======= Hello ========")
token = Whitespace()
word =  (token.tokenize("This is Pidor モバイルドメインリストの最新版が欲しい"))

# print(dir(word))
#
# print(dir(word))
# for e in word:
#     print(e.text)
#     print(dir(e.text))

