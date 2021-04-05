# batch_tokenize_openwebtext.py
# author: Diego Magdaleno.
# Take the bytepairencoding.py module and use it to tackle the
# openwebtext corpus in chunks (rather than all at once). An Encoder
# object from the module will be used to learn and contain the
# vocabulary from sections of the corpus. These Encoder objects will
# then be brought together into one Encoder object, storing not
# necessarily the full vocabulary of the openwebtext corpus, but rather
# the whole collection of (subword) tokens created from all sub
# Encoders. From there, this master Encoder can be used to tokenize the
# texts in the corpus and prepare it for training on a GPT2 object for
# what I hope to be a close match to the original created by OpenAI but
# is still useable on a "standard" computer/system.
# Python 3.7
# Windows/MacOS/Linux


import os
import gc
import re
import json
import string
import collections
import multiprocessing as mp
#import gpt2
import tensorflow as tf
#from tensorflow.keras.preprocessing.text import Tokenizer
#from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
#import tensorflow_datasets as tfds
import bytepairencoding as bpe
from datetime import datetime
from tqdm import tqdm


#'''
# Configuration code for allowing GPU usage on Tensorflow 2. Comment
# out when running on Tensorflow 1 on CPU.
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
session = tf.compat.v1.Session(config=config)
#'''


def main():
	# Get the list of files in the openwebtext corpus folder.
	path = "./openwebtext/"
	files = [path + file for file in os.listdir(path)]
	FILES_PER_CHUNK = 5000
	#FILES_PER_CHUNK = 50

	# Split the files up into sections of 5000 text files per chunk.
	# Given the total number of files, there will be 5 chunks with
	# the last one only containing about 600 files in it.
	file_blocks = list(divide_into_chunks(files, FILES_PER_CHUNK))

	# Dictionary of all learned tokens from the corpus.
	global_tokens = {}
	
	# Iterate through each section of files.
	for chunk in file_blocks:
		#print(len(chunk))
		print("On chunk {} of {}".format(file_blocks.index(chunk) + 1, len(file_blocks)))
		save_path = "./openwebtext_chunk_encoder{}/".format(file_blocks.index(chunk))

		# If the encoder was already created, load it. Otherwise,
		# read all the files listed in the section and store the
		# vocabulary to an Encoder object. Then learn the vocabulary.
		if os.path.exists(save_path) and len(os.listdir(save_path)) == 3:
			chunk_encoder = bpe.Encoder()
			chunk_encoder.load(save_path)
		else:
			# Get the vocabulary for all the files in the section.
			chunk_vocab = {}
			'''
			for text in chunk:
				local_vocab = bpe.get_vocab(text)
				chunk_vocab = bpe.combine_vocab(chunk_vocab, local_vocab)
			'''
			for text in tqdm(range(len(chunk))):
				local_vocab = bpe.get_vocab(chunk[text])
				chunk_vocab = bpe.combine_vocab(chunk_vocab, local_vocab)

			# Load that vocabulary into an Encoder object and learn it
			# for either a set number of merges or until the number of
			# tokens has reached a maximum. Save the details of that
			# Encoder.
			chunk_encoder = bpe.Encoder(chunk_vocab)
			chunk_encoder.learn_vocab(num_merges=2**14)
			#chunk_encoder.learn_vocab(num_merges=2**10)
			chunk_encoder.save(save_path)

		# Add the tokens to the global dictionary of learned tokens.
		global_tokens = bpe.combine_vocab(global_tokens, chunk_encoder.tokens)

		# Delete the Encoder object and call Python's garbage
		# collection to help remove any unused memory.
		del chunk_encoder
		gc.collect()

	# Save the global dictionary of all tokens learned in the corpus.
	with open("openwebtext_corpus_tokens.json", "w+", encoding="utf-8") as token_file:
		json.dump(global_tokens, token_file)

	# Exit the program.
	exit(0)


# Divide a list of items into n sized chunks.
# @param: list_in, a list of items that are to be separated into 
#	chunks.
# @param: chunk_size, the size each chunk is meant to be.
# @return: returns a list of lists, where each list contains an n sized
#	subset of the original list items.
def divide_into_chunks(list_in, chunk_size):
	for i in range(0, len(list_in), chunk_size):
		yield list_in[i:i + chunk_size]


if __name__ == '__main__':
	main()