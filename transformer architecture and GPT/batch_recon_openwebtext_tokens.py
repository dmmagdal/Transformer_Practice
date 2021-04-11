# batch_recon_openwebtext_tokens.py
# author: Diego Magdaleno.
# Get an idea of the number of unique tokens per chunk of texts from
# the openwebtext corpus. This will help figure things out for
# batch_tokenize_openwebtext.py.
# Python 3.7
# Windows/MacOS/Linux


import os
import gc
import re
import json
import string
import collections
import multiprocessing as mp
import bytepairencoding as bpe
from datetime import datetime
from tqdm import tqdm


def main():
	# Get the list of files in the openwebtext corpus folder.
	path = "./openwebtext/"
	files = [path + file for file in os.listdir(path)]
	FILES_PER_CHUNK = 1000

	# Split the files up into sections of 5000 text files per chunk.
	# Given the total number of files, there will be 5 chunks with
	# the last one only containing about 600 files in it.
	file_blocks = list(divide_into_chunks(files, FILES_PER_CHUNK))

	# Dictionary of all learned tokens from the corpus.
	global_tokens = {}
	assumed_tokens_count = 0
	global_tokens = set()
	
	# Iterate through each section of files.
	for chunk in file_blocks:
		#print(len(chunk))
		print("On chunk {} of {}".format(file_blocks.index(chunk) + 1, len(file_blocks)))

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

		# Get the tokens from the chunk.
		chunk_tokens = bpe.get_tokens(chunk_vocab)
		print("Number of items in chunk vocabulary:{}".format(len(chunk_vocab)))
		print("Number of tokens on chunk:{}".format(len(chunk_tokens)))
		assumed_tokens_count += len(chunk_tokens)

		# Add the tokens to the global dictionary of learned tokens.
		#global_tokens = bpe.combine_vocab(global_tokens, chunk_encoder.tokens)
		for k in list(chunk_tokens.keys()):
			global_tokens.add(k)

		# Delete the Encoder object and call Python's garbage
		# collection to help remove any unused memory.
		# del chunk_encoder
		# gc.collect()
	print("Total number of assumed tokens {}".format(assumed_tokens_count))
	print("Total number of actual unique tokens {}".format(len(global_tokens)))

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