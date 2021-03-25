# gutenbergtest.py
# author: Diego Magdaleno
# Test the bytepairencoding.py functions on the project gutenberg
# example outlined here in https://leimao.github.io/blog/
# Byte-Pair-Encoding/.
# Python 3.7


import re
import collections
import bytepairencoding as bpe


def main():
	initial_vocab = bpe.get_vocab("pg16457.txt")
	initial_tokens = bpe.get_tokens(initial_vocab)

	pairs = bpe.get_stats(initial_vocab)
	best = max(pairs, key=pairs.get)
	print(best)
	new_vocab = bpe.merge_vocab(best, initial_vocab)
	new_tokens = bpe.get_tokens(new_vocab)

	print(new_tokens == initial_tokens)
	print(new_vocab == initial_vocab)


def get_vocab(filepath):
	vocab = collections.defaultdict(int)
	with open(filepath, "r", encoding="utf-8") as read_file:
		for line in read_file:
			words = line.strip().split()
			for word in words:
				vocab[" ".join(list(word)) + " </w>"] += 1
	return vocab


def get_stats(vocab):
	pairs = collections.defaultdict(int)
	for word, freq in vocab.items():
		symbols = word.split()
		for i in range(len(symbols) - 1):
			pairs[symbols[i], symbols[i + 1]] += freq
	return pairs


def merge_vocab(pair, in_vocab):
	out_vocab = {}
	bigram = re.escape(" ".join(pair))
	p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
	for word in in_vocab:
		out_word = p.sub("".join(pair), word)
		out_vocab[out_word] = in_vocab[word]
	return out_vocab


def get_tokens(vocab):
	tokens = collections.defaultdict(int)
	for word, freq in vocab.items():
		word_tokens = word.split()
		for token in word_tokens:
			tokens[token] += freq
	return tokens


if __name__ == '__main__':
	main()