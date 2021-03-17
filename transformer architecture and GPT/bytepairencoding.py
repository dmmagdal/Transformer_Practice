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


def get_vocab(file_path):
	# Initialize an empty dictionary to hold the vocabulary (maps words
	# found in the text to the number of times they occur in the file,
	# also called its frequency).
	vocab = collections.defaultdict(int)
	with open(file_path, "r", encoding="utf-8") as open_file:
		for line in open_file:
			words = line.strip().split()
			for word in words:
				vocab[" ".join(list(word) + "</w>")] += 1

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


def merge_vocab(pair, new_vocab):
	ouptut_vocab = {}
	bigram = re.escape(" ".join(pair))
	p = re.compile(r'(?<!\S)' + bigram + r'(?<!\S)')
	for word in new_vocab:
		output_word = p.sub("".join(pair), word)
		output_vocab[output_word] = new_vocab[output_word]
	return output_vocab


def get_tokens(vocab):
	tokens = collections.defaultdict(int)
	for word, freq in vocab.items():
		word_tokens = word.split()
		for token in word_tokens:
			tokens[token] += freq
	return tokens


# Create a loop that will create tokenization out of the vocabulary.
for i in range(num_merges):
	pairs = get_stats(vocab)

	if not pairs:
		break

	best = max(pairs, key=pairs.get)
	vocab = merge_vocab(best, vocab)
	tokens = get_tokens(vocab)


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


def tokenize_word(string, sorted_tokens, unkown_token="</u>"):
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


tokens_frequencies, vocab_tokenization = get_tokens_from_vocab(vocab)
sorted_tokens_tuple = sorted(tokens_frequencies.items(), key=lambda item: (measure_token_length(item[0]), item[1]), reverse=True)
sorted_tokens = [token for (token, freq) in sorted_tokens_tuple]

