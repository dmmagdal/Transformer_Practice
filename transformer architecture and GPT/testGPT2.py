# testGPT2.py
# author: Diego Magdaleno
# This program tests the gpt2.py module I wrote with tensorflow. The
# primary function of the gpt2.py model trained will be text
# generation.
# Sources:
# https://keras.io/examples/generative/text_generation_with_miniature_gpt/
# Python 3.7
# Tensorflow 1.14/1.15/2.4
# Windows/MacOS/Linux


import os
import string
import random
import re
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import tensorflow_datasets as tfds
import gpt2


#'''
# Configuration code for allowing GPU usage on Tensorflow 2. Comment
# out when running on Tensorflow 1 on CPU.
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
session = tf.compat.v1.Session(config=config)
#'''


def main():
	# Model hyperparameters.
	vocab_size = 20000 # Only consider the top 20K words.
	max_len = 80 # Maximum sequence size.
	embedding_size = 256 # Embedding size for each token.
	n_heads = 2 # Number of attention heads.
	ff_dim = 256 # Hidden layer size in feed forward neural network
	# inside transformer.

	# OpenAI GPT-2 Hyperparameters
	# Model name 		embed_dim	n_layers	n_heads
	# Small (124M)		768				12			12
	# Medium (355M)		1024			24			16
	# Large (774M)		1280			36			20
	# X-Large (1.5B)	1600			48			25
	# GPT-2 max_len or context_size = 1024
	# GPT-2 vocab_size = 50257

	# Load in the dataset. Data is the IMDB dataset and can be found
	# at https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
	# and unpacked with "tar -xf aclImdb_v1.tar.gz".
	batch_size = 128

	# The data set contains each review in a separate text file. The
	# text files are present in four different folders. Create a list
	# of all files.
	filenames = []
	directories = [
		"aclImdb/train/pos",
		"aclImdb/train/neg",
		"aclImdb/test/pos",
		"aclImdb/test/neg"
	]
	for d in directories:
		for f in os.listdir(d):
			filenames.append(os.path.join(d, f))
	print(f"{len(filenames)} files")

	# Create a dataset from the text fiels.
	random.shuffle(filenames)
	text_ds = tf.data.TextLineDataset(filenames)
	text_ds = text_ds.shuffle(buffer_size=256)
	text_ds = text_ds.batch(batch_size)

	# Remove all html line-break tags and handle punctuation.
	# @param: input_string, the input string of the data.
	# @return: returns a cleaned version of the string.
	def custom_standardization(input_string):
		lowercased = tf.strings.lower(input_string)
		stripped_html = tf.strings.regex_replace(lowercased, "<br />", " ")
		return tf.strings.regex_replace(stripped_html, f"([{string.punctuation}])", r" \1")

	def prepare_lm_inputs_labels(text):
		text = tf.expand_dims(text, -1)
		tokenized_sentences = vectorize_layer(text)
		x = tokenized_sentences[:, :-1]
		y = tokenized_sentences[:, 1:]
		return x, y

	# Create a vectorization layer and adapt it to the text.
	vectorize_layer = TextVectorization(
		standardize=custom_standardization,
		max_tokens=vocab_size - 1,
		output_mode="int",
		output_sequence_length=max_len + 1
	)
	vectorize_layer.adapt(text_ds)
	vocab = vectorize_layer.get_vocabulary() # Get words back from token indices
	text_ds = text_ds.map(prepare_lm_inputs_labels)
	text_ds = text_ds.prefetch(tf.data.experimental.AUTOTUNE)

	# Training hyperparameters.
	verbose = 2
	epochs = 25

	# Tokenize starting prompt.
	word_to_index = {}
	for index, word in enumerate(vocab):
		word_to_index[word] = index
	start_prompt = "this movie is"
	start_tokens = [word_to_index.get(_, 1) for _ in start_prompt.split()]
	num_tokens_generated = 40
	text_gen_callback = gpt2.TextGenerator(num_tokens_generated, start_tokens, 
											vocab, max_len)

	# Initialize a GPT2 object.
	#new_gpt = gpt2.GPT2()
	new_gpt = gpt2.GPT2(n_heads=n_heads, n_layers=1, vocab_size=vocab_size,
						ff_dim=ff_dim, embedding_size=embedding_size, 
						context_size=max_len)

	# Train GPT2 on data.
	'''
	new_gpt.train_model(text_ds, batch_size=batch_size, verbose=verbose, 
						epochs=epochs, callbacks=[text_gen_callback])
	'''
	new_gpt.gpt_model.fit(text_ds, verbose=2, epochs=25, 
							callbacks=[text_gen_callback])

	'''
	# Run text generation with prompt.
	max_len = 50
	top_k = 5
	num_responses = 5
	prompt_1 = ""
	prompt_1 "Welcome my son, "
	#print(new_gpt.generate(prompt_1, max_len, temp, top_k, num_responses))
	#print(new_gpt.generate(prompt_2, max_len, temp, top_k, num_responses))
	'''

	# Save GPT2 model.
	save_path = "./test_gpt2_model_small"
	if not os.path.exists(save_path):
		os.mkdir(save_path)
	new_gpt.save(save_path)

	# Load GPT2 model.
	load_gpt = gpt2.GPT2()
	load_gpt.load(save_path)
	print(load_gpt.gpt_model.summary())
	
	# Exit the program.
	exit(0)


if __name__ == '__main__':
	main()