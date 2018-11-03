# -*- coding: utf-8 -*-

from rasa_nlu.training_data import load_data
from rasa_nlu import config
from rasa_nlu.model import Trainer
from rasa_nlu.model import Metadata, Interpreter

def train_nlu(data, configs, model_dir):
	training_data = load_data(data)
	trainer = Trainer(config.load(configs))
	trainer.train(training_data)
	trainer.persist(model_dir, fixed_model_name = 'jp_data')

def run_nlu():
	interpreter = Interpreter.load('./models/nlu/default/jp_data')

if __name__ == '__main__':
	train_nlu('./data/ja_data.json', 'config.yaml', './models/jp_data')

	# interpreter = Interpreter.load('./models/jp_data/default/jp_data')
	# print(interpreter)
	# print(dir(interpreter))
	# print(interpreter.pipeline)
	# print("Starting Bots")
	# while(True):
	# 	texts = raw_input(">>: ")
	# 	texts = unicode(texts, "utf-8")
	# 	# print(texts)
	# 	# print(type(texts))
	# 	print (interpreter.parse(texts))