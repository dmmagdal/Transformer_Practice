# bytepairencoding.py
# Create byte pair encodings of a text. Byte pair encoding (BPE) has
# an advantage in knowing how to deal with unknown words and can infer
# meaning in new words it encounters.
# Source: https://medium.com/@KaanBursa/byte-pair-encoding-21de9feb7a6d
# Tensorflow 1.14/1.15/2.4
# Python 3.7
# Windows/MacOS/Linux


import os
import re
import collections
import json
from tqdm import tqdm


def get_vocab(file_path):
	# Initialize an empty dictionary to hold the vocabulary (maps words
	# found in the text to the number of times they occur in the file,
	# also called its frequency).
	vocab = collections.defaultdict(int)
	with open(file_path, "r", encoding="utf-8") as open_file:
		for line in open_file:
			words = line.strip().split()
			for word in words:
				#vocab[" ".join(list(word)) + " </w>"] += 1

				# Alternative way to parse texts without marking the
				# end of each word with "</w>". Using this will affect
				# the original code for measure_token_length.
				vocab[" ".join(list(word))] += 1

	# Return the vocabulary dictionary.
	return vocab


def get_stats(vocab):
	# Initialize an empty dictionary to hold the pairs of characters
	# (maps character pairs to the frequency in which they occur in the
	# vocabulary).
	pairs = collections.defaultdict(int)
	for word, freq in vocab.items():
		symbols = word.split()
		for i in range(len(symbols) - 1):
			pairs[symbols[i], symbols[i + 1]] += freq

	# Return the character pairs dictionary.
	return pairs


def merge_vocab(pair, input_vocab):
	output_vocab = {}
	bigram = re.escape(" ".join(pair))
	p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
	for word in input_vocab:
		output_word = p.sub("".join(pair), word)
		output_vocab[output_word] = input_vocab[word]
	return output_vocab


def get_tokens(vocab):
	tokens = collections.defaultdict(int)
	for word, freq in vocab.items():
		word_tokens = word.split()
		for token in word_tokens:
			tokens[token] += freq
	return tokens


'''
# Create a loop that will create tokenization out of the vocabulary.
for i in range(num_merges):
	pairs = get_stats(vocab)

	if not pairs:
		break

	best = max(pairs, key=pairs.get)
	vocab = merge_vocab(best, vocab)
	tokens = get_tokens(vocab)
'''


# To encode a given sentence:
# 1) convert token dictionary from longest word to shortest word.
# 2) add split each word in the sentence and add <\w> to the end of the
# word.
# 3) iterate through each token and if the substring of the word
# includes the token we put that token as tokenization process.
# To decode a given sentence:
# 1) give the tokens, merge the word that does not have <\w> at the end
# and add "" if the word has "<\w>" at the end.
def get_tokens_from_vocab(vocab):
	tokens_frequencies = collections.defaultdict(int)
	vocab_tokenization = {}

	for word, freq in vocab.items():
		word_tokens = word.split()
		for token in word_tokens:
			tokens_frequencies[token] += freq
		vocab_tokenization["".join(word_tokens)] = word_tokens
	return tokens_frequencies, vocab_tokenization


def measure_token_length(token):
	if token[-4:] == "</w>":
		return len(token[:-4]) + 1
	else:
		return len(token)


def tokenize_word(string, sorted_tokens, unknown_token="</u>"):
	if string == "":
		return []

	if sorted_tokens == []:
		return [unknown_token]

	string_tokens = []
	for i in range(len(sorted_tokens)):
		token = sorted_tokens[i]
		token_reg = re.escape(token.replace(".", "[.]"))

		matched_positions = [(m.start(0), m.end(0)) for m in re.finditer(token_reg, string)]
		if len(matched_positions) == 0:
			continue
		substring_end_positions = [matched_position[0] for matched_position in matched_positions]

		substring_start_position = 0
		for substring_end_position in substring_end_positions:
			substring = string[substring_start_position:substring_end_position]
			string_tokens += tokenize_word(string=substring, sorted_tokens=sorted_tokens[i + 1:], unknown_token=unknown_token)
			string_tokens += [token]
			substring_start_position = substring_end_position + len(token)
		remaining_substring = string[substring_start_position:]
		string_tokens += tokenize_word(string=remaining_substring, sorted_tokens=sorted_tokens[i + 1:], unknown_token=unknown_token)
		break
	return string_tokens


'''
tokens_frequencies, vocab_tokenization = get_tokens_from_vocab(vocab)
sorted_tokens_tuple = sorted(tokens_frequencies.items(), key=lambda item: (measure_token_length(item[0]), item[1]), reverse=True)
sorted_tokens = [token for (token, freq) in sorted_tokens_tuple]
'''


# XXX --- BEGIN NOTE --- XXX
# All code implmented below this point is not from the source listed in
# the header.
# XXX --- END NOTE --- XXX


# Combine two vocab dictionaries into one (different function than
# merge_vocab and is not a part of the original code from the source
# listed in the header).
# @param: vocab1, a dictionary containing one vocabulary with its
#	mapping of words to frequencies.
# @param: vocab2, a dictionary containing another vocabulary with its
#	own mapping of words to frequencies.
# @return: returns a dictionary containing the combined words from both
#	vocabularies with the total frequencies.
def combine_vocab(vocab1, vocab2):
	# Intialize a new dictionary to store the merged vocabularies.
	new_vocab = {}

	# Store the words and their frequencies from the first vocabulary.
	#for new_word1, freq1 in vocab1.items():
	#	new_vocab[new_word1] = freq1
	new_vocab = vocab1

	# Store the words and their frequencies from the second vocabulary.
	# If a word already exists in the combined vocabulary (it exists in
	# the first vocabulary) then add the frequencies. Otherwise, the
	# the word is unique to the second vocabulary, so its frequency is
	# simply used.
	for new_word2, freq2 in vocab2.items():
		if new_word2 in new_vocab:
			new_vocab[new_word2] += freq2
		else:
			new_vocab[new_word2] = freq2

	# Return the combined vocabulary.
	return new_vocab


class Encoder:
	def __init__(self, vocab=None, tokens=None, tokens_to_value=None, learn_vocab=False, token_size=None, num_merges=1000):
		# Load in the vocab and tokens dictionaries from the arguments.
		# If they are not passed in (defualt None), initialize them to
		# empty dictionaries.
		if vocab:
			self.vocab = vocab
		else:
			self.vocab = collections.defaultdict(int)
		if tokens:
			self.tokens = tokens
		else:
			self.tokens = collections.defaultdict(int)
		if tokens_to_value:
			self.tokens_to_value = tokens_to_value
		else:
			self.tokens_to_value = collections.defaultdict(int)

		# If the learn_vocab argument was passed in as True, call the
		# function to merge the vocab and create the most frequent
		# subword tokens.
		if learn_vocab:
			self.learn_vocab(token_size=token_size, num_merges=num_merges)

		# Update the other class variables.
		self.update()
		'''
		# Use the existing vocabulary to get the tokens (and their
		# frequencies) and use that to create a sorted list of tokens.
		self.tokens_frequencies, self.vocab_tokenization = get_tokens_from_vocab(self.vocab)
		self.sorted_tokens_tuple = sorted(self.tokens_frequencies.items(), 
											key=lambda item: (measure_token_length(item[0]), item[1]), 
											reverse=True)
		self.sorted_tokens = [token for (token, freq) in self.sorted_tokens_tuple]
		'''


	# Perform the following loop using the vocabulary and tokens to
	# find the most frequent character pairs and create subword tokens
	# to store.
	# @param: token_size, the maximum number of tokens that are allowed
	#	to exist. Default value is None but this is usually an int 
	#	value.
	# @param: num_merges, the number of merges that are going to occur.
	#	Default value is 1000.
	# @return: returns nothing.
	def learn_vocab(self, token_size=None, num_merges=1000):
		# Get the smallest of either the maximum number of tokens to
		# have (token_size) or the number of merges (num_merges).
		if not token_size:
			merges = num_merges
		else:
			merges = max(token_size, num_merges)

		# Iterate through the following loop to update the vocabulary
		# (and tokens).
		print("Learning Vocabulary:")
		#for i in range(merges):
		for i in tqdm(range(merges)):
			# Break out of the loop if the token limit has been
			# initialized and is reached.
			if token_size and len(self.tokens) == token_size:
				break

			# Get the character pairs from the vocabulary and isolate
			# the most frequent character pair. 
			pairs = get_stats(self.vocab)
			if not pairs:
				break
			best = max(pairs, key=pairs.get)

			# Merge the vocabulary with the most frequent character
			# pair and update the tokens.
			self.vocab = merge_vocab(best, self.vocab)
			self.tokens = get_tokens(self.vocab)

		# Update the class variables.
		self.update()

		# Return the function.
		return


	# Convert a word into tokens and then convert those tokens to their
	# unique values and return that as a list.
	# @param: input_word, a word string that needs to be encoded.
	# @return:
	def encode(self, input_word):
		return tokenize_word(input_word, self.sorted_tokens, "</u>")


	# Convert a list of tokens and reconstruct a word.
	# @param: token_list, 
	# @return:
	def decode(self, token_list):
		return "".join(token_list)


	# Save the files for the encoder to the specified save folder path.
	# @param: save_folder_path,
	# @return: returns nothing.
	def save(self, save_folder_path):
		# If the save folder path does not exist or is not a directory,
		# create the directory at the specified path.
		if not os.path.exists(save_folder_path) or not os.path.isdir(save_folder_path):
			#print("Folder path " + save_folder_path + " could not be found.")
			#return
			os.mkdir(save_folder_path)

		# Open the vocabulary and tokens json files and save the data
		# to the respective variables.
		with open(save_folder_path + "vocab.json", "w+", encoding="utf-8") as vocab_file:
			json.dump(self.vocab, vocab_file, indent=4)
		with open(save_folder_path + "tokens.json", "w+", encoding="utf-8") as token_file:
			json.dump(self.tokens, token_file, indent=4)
		with open(save_folder_path + "token2value.json", "w+", encoding="utf-8") as token2value_file:
			json.dump(self.tokens_to_value, token2value_file, indent=4)

		# Return the function.
		return


	# Load in the files for the encoder from the specified save folder
	# path. Note that this will override the class variables.
	# @param: save_folder_path,
	# @return: returns nothing.
	def load(self, save_folder_path):
		# If the save folder path does not exist or is not a directory,
		# print and error message and return the function early. Repeat
		# this for all required save files that should exist in that
		# directory.
		if not os.path.exists(save_folder_path) or not os.path.isdir(save_folder_path):
			print("Folder path " + save_folder_path + " could not be found.")
			return
		if not os.path.exists(save_folder_path + "vocab.json") or not os.path.isfile(save_folder_path + "vocab.json"):
			print("File path " + save_folder_path + "vocab.json could not be found.")
			return
		if not os.path.exists(save_folder_path + "tokens.json") or not os.path.isfile(save_folder_path + "tokens.json"):
			print("File path " + save_folder_path + "tokens.json could not be found.")
			return
		if not os.path.exists(save_folder_path + "token2value.json") or not os.path.isfile(save_folder_path + "token2value.json"):
			print("File path " + save_folder_path + "token2value.json could not be found.")
			return

		# Open the vocabulary and tokens json files and save the data
		# to the respective variables.
		with open(save_folder_path + "vocab.json", "r", encoding="utf-8") as vocab_file:
			self.vocab = json.load(vocab_file)
		with open(save_folder_path + "tokens.json", "r", encoding="utf-8") as token_file:
			self.tokens = json.load(token_file)
		with open(save_folder_path + "token2value.json", "r", encoding="utf-8") as token2value_file:
			self.tokens_to_value = json.load(token2value_file)

		# Update the class variables given the loaded vocabulary and
		# tokens.
		self.update()

		# Return the function.
		return


	'''
	# XXX --- TODO --- XXX
	# Figure out how updating the vocabulary with new words and
	# frequencies will affect and update the remaining variables such
	# as the tokens.
	# XXX --- END TODO --- XXX
	# Update the current vocabulary.
	# @param: new_vocab, a dictionary mapping new vocabulary words to
	#	their frequency.
	# @return: returns nothing.
	def update_vocab(self, new_vocab):
		# Iterate through every new word in the new vocabulary. If the
		# new word already exists in the current vocabulary, add the
		# frequencies to the entry. Otherwise, the new word does not
		# exist in the vocabulary so the new word's frequency is used.
		for new_word, freq in new_vocab.items():
			if new_word in self.vocab:
				self.vocab[new_word] += freq
			else:
				self.vocab[new_word] = freq

		# Update the tokens and other class variables.
		#self.update()

		# Return the function.
		return
	'''


	# Update class variables other than the vocabulary.
	# @param: takes no arguments.
	# @return: returns nothing.
	def update(self):
		# Update the tokens from the vocabulary.
		self.tokens = get_tokens(self.vocab)

		# Update the tokens to values dictionary.
		value = 1
		for token in sorted(list(self.tokens.keys())):
			self.tokens_to_value[token] = value
			value += 1
		'''
		self.tokens_to_value[" "] = 0
		self.tokens_to_value["<|pad|>"] = value + 1
		self.tokens_to_value["<|endoftext|>"] = value + 2
		'''

		# Use the existing vocabulary to get the tokens (and their
		# frequencies) and use that to create a sorted list of tokens.
		self.tokens_frequencies, self.vocab_tokenization = get_tokens_from_vocab(self.vocab)
		self.sorted_tokens_tuple = sorted(self.tokens_frequencies.items(), 
											key=lambda item: (measure_token_length(item[0]), item[1]), 
											reverse=True)
		self.sorted_tokens = [token for (token, freq) in self.sorted_tokens_tuple]
		
		# Return the function.
		return