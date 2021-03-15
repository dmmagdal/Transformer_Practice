# tokenize_openwebtext.py
# author: Diego Magdaleno
# Tokenize and parse all texts from the openwebtext corpus dataset
# (extracted from the .tar.xz).
# Python 3.7
# Windows/MacOS/Linux


import os
import re
import string
import nltk
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import tensorflow_datasets as tfds


#'''
# Configuration code for allowing GPU usage on Tensorflow 2. Comment
# out when running on Tensorflow 1 on CPU.
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
session = tf.compat.v1.Session(config=config)
#'''


def main():
	# Retrieve the vocabulary set and the length of the longest line
	# from the openwebtext corpus.
	#openwebtext_vocab, longest_line_len = get_vocab_from_texts()
	# Use this value for faster runs. This value came from running the
	# get_vocab_from_texts() function on the openwebtext corpus dataset
	# using only the first 5,000 texts (out of 20,000 total).
	openwebtext_vocab = 10438152

	# Initialize a batch size and sequence length.
	max_seq_len = 1024
	batch_size = 128

	# Clean the files and convert the data to a dataset.
	path = "./openwebtext/"
	text_list = [path + file for file in os.listdir(path)]
	openwebtext_dataset = tf.data.TextLineDataset(text_list)
	openwebtext_dataset = openwebtext_dataset.shuffle(buffer_size=256)
	openwebtext_dataset = openwebtext_dataset.batch(batch_size)
	text_vector = TextVectorization(
					# standardize=custom_standardization,
					# max_tokens=len(openwebtext_vocab) - 1,
					max_tokens=openwebtext_vocab - 1,
					output_mode="int",
					output_sequence_length=max_seq_len + 1)
	text_vector.adapt(openwebtext_dataset)
	vocab = text_vector.get_vocabulary()

	# Shift words by 1 position so that the target for position (i) is
	# word at position (i + 1). The model will use all words up till
	# position (i) to predict the next word.
	# @param: text, the text from a text dataset.
	# @return: returns a tuple of the input and output tokenized 
	#	sequences for the text provided.
	def prepare_input_labels(text):
		text = tf.expand_dims(text, -1)
		tokenized_sentences = text_vector(text)
		x = tokenized_sentences[:, :-1]
		y = tokenized_sentences[:, 1:]
		return x, y

	openwebtext_dataset = openwebtext_dataset.map(prepare_input_labels)
	openwebtext_dataset = openwebtext_dataset.prefetch(tf.data.experimental.AUTOTUNE)

	'''
	# Intiailize the text encoders.
	byte_encoder = tfds.deprecated.text.ByteTextEncoder()
	subword_encoder = tfds.deprecated.text.SubwordTextEncoder(vocab_list=list(openwebtext_vocab))

	# Save the text encoders.
	byte_encoder.save_to_file("./byte_encoder")
	subword_encoder.save_to_file("./subword_encoder")

	# Load the text encoders.
	byte_encoder.load_from_file("./byte_encoder")
	subword_encoder.load_from_file("./subword_encoder")
	'''

	# Exit the main program.
	exit(0)


# Load in all texts from the specified path and tokenize them with the
# TextVectorization layer from tensorflow/keras to create a global
# vocabulary from those texts.
# @param: texts_folder, a string that contains the path to the folder
#	containing all texts to be trained on. By default, that value is
#	for the openwebtext folder created from the 
#	openwebtext_uncompress.py
# @param: limit, an int value that issues how many texts to stop at. BY
#	default the value None (which allows for all texts to be read and
#	parsed).
# @return: returns the global vocabulary set created from all texts.
def get_vocab_from_texts(texts_folder="./openwebtext/", limit=None):
	# Read through each text and figure out the total (as well as
	# local) vocab size.
	text_files = os.listdir(texts_folder)

	# Analyze and create a vocabulary from all text files.
	'''
	global_vector = TextVectorization()
	global_file_lines = []
	'''
	global_vocab = set()
	max_line_len = 0
	for file in text_files:
		# with open("./openwebtext/" + file, "rb") as open_file:
		with open(texts_folder + file, "r", encoding="utf-8") as open_file:
			file_contents = open_file.read()
		#print(type(file_contents))
		file_lines = file_contents.split("\n")

		if texts_folder == "./openwebtext/":
			cleaned_lines = clean_file_lines(file_lines)
		else:
			cleaned_lines = file_lines
		# Iterate through the resuling cleaned lines and get the length
		# of the line (length of the string, not the same as the number
		# of tokens that make up the string).
		for line in cleaned_lines:
			if len(line) > max_line_len:
				max_line_len = len(line)

		# Analyze and create a vocabulary from the text file.
		text_vector = TextVectorization()
		text_vector.adapt(cleaned_lines)
		vocab = text_vector.get_vocabulary()
		print("Local vocab size: {}".format(len(vocab)))

		# Merge vocabulary with the global vocabulary set. (Note: Using
		# set union of other vocabulary is the most memory efficient vs
		# adapting another TextVectorization layer to the entirety of
		# all texts).
		#global_file_lines += file_lines
		global_vocab = global_vocab.union(set(vocab))

		# Break the loop if a limit was passed in and has been reached.
		if limit is not None and text_files.index(file) == limit:
			break

		# (FOR DEBUG. COMMENT OUT FOR PRODUCTION)
		# Repeat loop for only 5,000 texts (out of 20,000). Anymore
		# causes the program to run out of memory for the openwebtext
		# corpus dataset.
		if text_files.index(file) == 5000:
			break

	# Print the size of the global vocabulary.
	'''
	global_vector.adapt(global_file_lines)
	print("Global vocab size: {}".format(len(global_vector.get_vocabulary())))
	'''
	print("Global vocab size: {}".format(len(global_vocab)))
	print("Longest line length in corpus: {} characters".format(max_line_len))

	# Return the global vocabulary set along with the longest length
	# of a line from all texts.
	return global_vocab, max_line_len


# Clean the list of lines from a file such that special strings (such
# as the long one found in openwebtext text files) are removed from the
# list of lines contained in a file.
# @param: file_lines, a list of strings representing all lines of text
#	from a file.
# @return: returns a modified copy of the list argument that is 
#	"cleaned".
def clean_file_lines(file_lines):
	# There are special strings in the text. These may interfere
	# with tokenization/encoding and need to be removed.
	# Examples:
	# 0107725-5c1cfcbb66068a2e76d1b7d3350adc2a.txt0000644000000000000000000002364700000000000015324 0ustar  00000000000000
	# 0107919-6e84ed348c613c3f24ac6999d4fdbcfc.txt0000644000000000000000000000705000000000000015362 0ustar  00000000000000
	# String length is 116. Magic string value in all is the
	# "ustar". Check for these special strings in a line. Use the
	# "\x00" to guage where to splice the string. This will remove
	# both the "\x00" and the long unique string from the texts.
	for line in range(len(file_lines)):
		# Use list splicing to remove the special string if it's
		# found in that line.
		if "\x00" in file_lines[line] and "ustar" in file_lines[line]:
			start_index = file_lines[line].index("\x00")
			end_index = file_lines[line][::-1].index("\x00")
			file_lines[line] = file_lines[line][:start_index] + " " + file_lines[line][-end_index:]
		# If the line does not have the special string, but still
		# has the "\x00" value, just remove all instances of that
		# value from the line.
		elif "\x00" in file_lines[line]:
			file_lines[line] = file_lines[line].replace("\x00", "")

	# Remove empty lines that are just "\n"
	while "\n" in file_lines:
		file_lines.remove("\n")

	# Return the cleaned file lines.
	return file_lines


'''
# Standardize the texts in the dataset.
# @param: input_string, a tesnor from the text dataset.
# @return: returns the input_string modified
def custom_standardization(input_string):
	# There are special strings in the text. These may interfere
	# with tokenization/encoding and need to be removed. Check for
	# these special strings in a line. Use the "\x00" to guage where to
	# splice the string. This will remove both the "\x00" and the long
	# unique string from the texts.
	if "\x00" in input_string and "ustar" in input_string:
		start_index = input_string.index("\x00")
		end_index = input_string[::-1].index("\x00")
		return_string = input_string[:start_index] + " " + input_string[-end_index:]
	# If the line does not have the special string, but still
	# has the "\x00" value, just remove all instances of that
	# value from the line.
	elif "\x00" in input_string:
		return_string = input_string.replace("\x00", "")
	else:
		return_string = input_string

	# Return the modified input string.
	return return_string
'''


if __name__ == '__main__':
	main()