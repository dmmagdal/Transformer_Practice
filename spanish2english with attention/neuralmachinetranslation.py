# neuralmachinetranslation.py
# Train a sequence to sequence model for spanish to english translation
# using attention.
# source: https://www.tensorflow.org/tutorials/text/nmt_with_attention
# Python 3.7
# Tensorflow 1.14/1.15/2.40
# Windows/MacOS/Linux


import os
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split

import unicodedata
import re
import numpy as np
import io
import time


#'''
# Configuration code for allowing GPU usage on Tensorflow 2. Comment
# out when running on Tensorflow 1 on CPU.
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
session = tf.compat.v1.Session(config=config)
#'''


def main():
	# Download dataset provided by http://www.manythings.org/anki/.
	# Format for language translation pairs is:
	# May I borrow this book? ?Puede tomar prestado este libro?
	# The dataset can be downloaded from the google cloud as well.
	origin = "http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip"
	path_to_zip = tf.keras.utils.get_file(
		"spa-eng.zip", origin=origin, extract=True)
	path_to_file = os.path.dirname(path_to_zip) + "/spa-eng/spa.txt"

	# Convert the unicode file to ascii.
	def unicode_to_ascii(s):
		return "".join(c for c in unicodedata.normalize("NFD", s)
						if unicodedata.category(c) != "Mn")

	def preprocess_sentence(w):
		w = unicode_to_ascii(w.lower().strip())

		# Create a space between a work and the punctuation following
		# it. e.g. "He is a boy." => "he is a boy ."
		# Reference:- https://stackoverflow.com/questions/3645931/
		# python-padding-punctuation-with-white-spaces-keeping-
		# punctuation
		w = re.sub(r"([?.!,¿])", r" \1 ", w)
		w = re.sub(r'[" "]+', " ", w)

		# Replace everything with space except (a-z, A-Z, ".", "?",
		# "!", ",").
		w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)

		w = w.strip()

		# Adding a start and end token to the sentence so that the
		# model knows when to start and stop predicting.
		w = "<start> " + w + " <end>"
		return w

	en_sentence = u"May I borrow this book?"
	sp_sentence = u"¿Puedo tomar prestado este libro?"
	print(preprocess_sentence(en_sentence))
	print(preprocess_sentence(sp_sentence).encode("utf-8"))

	# Remove accents, clean the sentences, and return word paris in the
	# format: [ENGLISH, SPANISH].
	def create_dataset(path, num_examples):
		lines = io.open(path, encoding="UTF-8").read().strip().split("\n")
		word_pairs = [[preprocess_sentence(w) for w in l.split("\t")] for l in lines[:num_examples]]
		return zip(*word_pairs)

	en, sp = create_dataset(path_to_file, None)
	print(en[-1])
	print(sp[-1])

	def tokenize(lang):
		lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters="")
		lang_tokenizer.fit_on_texts(lang)

		tensor = lang_tokenizer.texts_to_sequences(lang)
		tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding="post")
		return tensor, lang_tokenizer

	def load_dataset(path, num_examples=None):
		# Create cleaned input, output pairs.
		target_lang, inp_lang = create_dataset(path, num_examples)

		input_tensor, inp_lang_tokenizer = tokenize(inp_lang)
		target_tensor, tar_lang_tokenizer = tokenize(target_lang)

		return input_tensor, inp_lang_tokenizer, target_tensor, tar_lang_tokenizer

	# Optional: Limit the size of the dataset to experiment faster.
	# Training on a dataset of >100,000 sentences will take a long
	# time. To train faster, we can limit the size of the dataset to
	# 30,000 sentences (of course, translation quality degrads with
	# less data).
	# Try experimenting with the size of the dataset.
	num_examples = 100000
	num_examples = 30000
	input_tensor, inp_lang, target_tensor, target_lang = load_dataset(path_to_file, num_examples)

	# Calculate the max_length of the target tensors.
	max_length_targ, max_length_input = target_tensor.shape[1], input_tensor.shape[1]

	# Create training and validation sets using an 80-20 split.
	input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor, target_tensor, test_size=0.2)

	# Show length.
	print(len(input_tensor_train), len(input_tensor_val), len(target_tensor_train), len(target_tensor_val))

	def convert(lang, tensor):
		for t in tensor:
			if t != 0:
				print("%d ----> %s" % (t, lang.index_word[t]))

	print("Input Language; index to word mapping")
	convert(inp_lang, input_tensor_train[0])
	print()
	print("Target Language; index to word mapping")
	convert(target_lang, target_tensor_train[0])

	# Create a tf.data dataset
	BUFFER_SIZE = len(input_tensor_train)
	BATCH_SIZE = 64
	steps_per_epoch = len(input_tensor_train) // BATCH_SIZE
	embedding_dim = 256
	units = 1024
	vocab_inp_size = len(inp_lang.word_index) + 1
	vocab_tar_size = len(target_lang.word_index) + 1

	dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
	dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

	example_input_batch, example_target_batch = next(iter(dataset))
	print(example_input_batch.shape, example_target_batch.shape)

	# Write the encoder decoder model.
	class Encoder(tf.keras.Model):
		def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
			super(Encoder, self).__init__()
			self.batch_sz = batch_sz
			self.enc_units = enc_units
			self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
			self.gru = tf.keras.layers.GRU(self.enc_units,
											return_sequences=True,
											return_state=True,
											recurrent_initializer="glorot_uniform")


		def call(self, x, hidden):
			x = self.embedding(x)
			output, state = self.gru(x, initial_state=hidden)
			return output, state


		def initialize_hidden_state(self):
			return tf.zeros((self.batch_sz, self.enc_units))


	encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)

	# Sample input
	sample_hidden = encoder.initialize_hidden_state()
	sample_output, sample_hidden = encoder(example_input_batch, sample_hidden)
	print("Encoder output shape: (batch_size, sequence_lengh, units) {}".format(sample_output.shape))
	print("Encoder hidden state shape: (batch_size, units) {}".format(sample_hidden.shape))

	class BahdanauAttention(tf.keras.layers.Layer):
		def __init__(self, units):
			super(BahdanauAttention, self).__init__()
			self.w1 = tf.keras.layers.Dense(units)
			self.w2 = tf.keras.layers.Dense(units)
			self.v = tf.keras.layers.Dense(1)


		def call(self, query, values):
			# Query hidden state shape == (batch_size, hidden_size)
			# query_with_time_axis shape == (batch_size, 1, hidden_size)
			# Values shape == (batch_size, max_len, hidden_size)
			# We are doing this to broadcast the addition along the
			# time axis to calculate the score.
			query_with_time_axis = tf.expand_dims(query, 1)

			# score shape == (batch_size, max_length, 1)
			# We get 1 at the last axis because twe are applying score
			# to self.v
			# The shape of the tensor before applying self.v is
			# (batch_size, max_length, units)
			score = self.v(tf.nn.tanh(
							self.w1(query_with_time_axis) + self.w2(values)))

			# attention_weights shape == (batch_size, max_length, 1)
			attention_weights = tf.nn.softmax(score, axis=1)

			# context_vector shape after num == (batch_size, hidden_size)
			context_vector = attention_weights * values
			context_vector = tf.reduce_sum(context_vector, axis=1)
			return context_vector, attention_weights

	attention_layer = BahdanauAttention(10)
	attention_result, attention_weights = attention_layer(sample_hidden, sample_output)
	print("Attention result shape: (batch_size, units) {}".format(attention_result.shape))
	print("Attention weights shape: (batch_size, sequence_lengh, 1) {}".format(attention_weights.shape))

	class Decoder(tf.keras.Model):
		def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
			super(Decoder, self).__init__()
			self.batch_sz = batch_sz
			self.dec_units = dec_units
			self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
			self.gru = tf.keras.layers.GRU(self.dec_units,
											return_sequences=True,
											return_state=True,
											recurrent_initializer="glorot_uniform")
			self.fc = tf.keras.layers.Dense(vocab_size)

			# Used for attention.
			self.attention = BahdanauAttention(self.dec_units)


		def call(self, x, hidden, enc_output):
			# enc_output shape == (batch_size, max_length, hidden_size)
			context_vector, attention_weights = self.attention(hidden, enc_output)

			# x shape after passing through embedding == (batch_size, 1, embedding_dim)
			x = self.embedding(x)

			# x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
			x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

			# passing the concatenation vector to the GRU
			output, state = self.gru(x)

			# output shape == (batch_size * 1, hidden_size)
			output = tf.reshape(output, (-1, output.shape[2]))

			# output shape == (batch_size, vocab)
			x = self.fc(output)
			return x, state, attention_weights


	decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE)
	sample_decoder_output, _, _ = decoder(tf.random.uniform((BATCH_SIZE, 1)),
											sample_hidden, sample_output)
	print("Decoder output shape: (batch_size, vocab_size) {}".format(sample_decoder_output.shape))

	# Define the optimizer and loss function.
	optimizer = tf.keras.optimizers.Adam()
	loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
					from_logits=True, reduction="none")

	def loss_function(real, pred):
		mask = tf.math.logical_not(tf.math.equal(real, 0))
		loss_ = loss_object(real, pred)

		mask = tf.cast(mask, dtype=loss_.dtype)
		loss_ *= mask
		return tf.reduce_mean(loss_)

	# Checkpoints.
	checkpoint_dir = "./training_checkpoints"
	checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
	checkpoint = tf.train.Checkpoint(optimizer=optimizer,
									encoder=encoder,
									decoder=decoder)

	# Training.
	@tf.function
	def train_step(inp, tar, enc_hidden):
		loss = 0

		with tf.GradientTape() as tape:
			enc_output, enc_hidden = encoder(inp, enc_hidden)

			dec_hidden = enc_hidden

			dec_input = tf.expand_dims([target_lang.word_index["<start>"]] * BATCH_SIZE, 1)

			# Teacher forcing - feeding the target as the next input.
			for t in range(1, tar.shape[1]):
				# passing enc_output to the decoder.
				predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)

				loss += loss_function(tar[:, t], predictions)

				# Using teacher forcing.
				dec_input = tf.expand_dims(tar[:, t], 1)

		batch_loss = (loss / int(tar.shape[1]))

		variables = encoder.trainable_variables + decoder.trainable_variables

		gradients = tape.gradient(loss, variables)

		optimizer.apply_gradients(zip(gradients, variables))

		return batch_loss

	EPOCHS = 10

	for epoch in range(EPOCHS):
		start = time.time()

		enc_hidden = encoder.initialize_hidden_state()
		total_loss = 0

		for (batch, (inp, tar)) in enumerate(dataset.take(steps_per_epoch)):
			batch_loss = train_step(inp, tar, enc_hidden)
			total_loss += batch_loss

			if batch % 100 == 0:
				print("Epoch {} Batch {} Loss {:.4f}".format(epoch + 1,
															batch,
															batch_loss.numpy()))

		# Saving (checkpoint) the model every 2 epochs.
		if (epoch + 1) % 2 == 0:
			checkpoint.save(file_prefix=checkpoint_prefix)

		print("Epoch {} Loss {:.4f}".format(epoch + 1,
											total_loss / steps_per_epoch))
		print("Time take for 1 epoch {} sec\n".format(time.time() - start))

	# Translate
	def evaluate(sentence):
		attention_plot = np.zeros((max_length_targ, max_length_input))

		sentence = preprocess_sentence(sentence)

		inputs = [inp_lang.word_index[i] for i in sentence.split(" ")]
		inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
																maxlen=max_length_input,
																padding="post")
		inputs = tf.convert_to_tensor(inputs)

		result = ""

		hidden = [tf.zeros((1, units))]
		enc_output, enc_hidden = encoder(inputs, hidden)

		dec_hidden = enc_hidden
		dec_input = tf.expand_dims([target_lang.word_index["<start>"]], 0)

		for t in range(max_length_targ):
			predictions, dec_hidden, attention_weights = decoder(dec_input,
																dec_hidden,
																enc_output)

			# Store the attention weights to plot later.
			attention_weights = tf.reshape(attention_weights, (-1, ))
			attention_plot[t] = attention_weights.numpy()

			predicted_id = tf.argmax(predictions[0]).numpy()

			result += target_lang.index_word[predicted_id] + " "

			if target_lang.index_word[predicted_id] == "<end>":
				return result, sentence, attention_plot

			# The predicted ID is fed back into the model.
			dec_input = tf.expand_dims([predicted_id], 0)

		return result, sentence, attention_plot

	# Function for plotting the attention weights.
	def plot_attention(attention, sentence, predicted_sentence):
		fig = plt.figure(figsize=(10,10))
		ax = fig.add_subplot(1, 1, 1)
		ax.matshow(attention, cmap="viridis")

		fontdict = {"fontsize": 14}

		ax.set_xticklabels([""] + sentence, fontdict=fontdict, rotation=90)
		ax.set_yticklabels([""] + predicted_sentence, fontdict=fontdict)
		plt.show()

	def translate(sentence):
		result, sentence, attention_plot = evaluate(sentence)

		print("Input: %s" % (sentence))
		print("Predicted translation: {}".format(result))

		attention_plot = attention_plot[:len(result.split(" ")), :len(sentence.split(" "))]
		plot_attention(attention_plot, sentence.split(" "), result.split(" "))

	# Restore the last checkpoint and test.
	checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
	translate(u"hace mucho frio aqui.")
	translate(u"esta es mi vida.")
	translate(u'¿todavia estan en casa?')

	# wrong translation
	translate(u'trata de averiguarlo.')


if __name__ == '__main__':
	main()