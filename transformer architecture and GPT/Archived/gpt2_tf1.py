# gpt2_tf1.py
# author: Diego Magdaleno
# A recreation of the GPT-2 model from OpenAI. This model is more of a
# conversion of the existing model to a tensorflow/keras model. Note
# that this version is meant to run on tensorflow 1.14/1.15 and does
# not use the MultiHeadAttention layer from tensorflow/keras because it
# did not exist until tensorflow 2.4.
# Sources:
# https://keras.io/examples/generative/text_generation_with_miniature_gpt/
# https://keras.io/api/models/model/
# https://www.tensorflow.org/api_docs/python/tf/compat/v1/keras/layers/experimental/preprocessing/TextVectorization
# https://towardsdatascience.com/you-should-try-the-new-tensorflows-textvectorization-layer-a80b3c6b00ee
# https://www.tensorflow.org/api_docs/python/tf/keras/layers/Embedding
# https://www.tensorflow.org/api_docs/python/tf/keras/layers/MultiHeadAttention
# https://github.com/tensorflow/tensorflow/blob/v2.4.1/tensorflow/python/keras/layers/multi_head_attention.py#L126-L479
# Python 3.7
# Tensorflow 1.14/1.15/2.4
# Windows/MacOS/Linux


import os
import json
import regex as re
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Conv1D, Embedding, Dropout
from tensorflow.keras.layers import LayerNormalization, Input
from tensorflow.keras.layers import Attention
import tensorflow_addons as tfa
from tensorflow_addons.layers import MultiHeadAttention
#import tensorflow_datasets as tfds


'''
# Configuration code for allowing GPU usage on Tensorflow 2. Comment
# out when running on Tensorflow 1 on CPU.
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
session = tf.compat.v1.Session(config=config)
'''

# Mask all the pad tokens in the batch of sequence. It ensures that
# the model does not treat padding as the input. The mask indicates
# where pad value 0 is present: it outputs a 1 at those locations
# and a 0 otherwise.
def create_padding_mask(seq):
	seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

	# Add extra dimensions to add the padding to the attention
	# logits.
	return seq[:, tf.newaxis, tf.newaxis, :] # (batch_size, 1, 1, seq_len)


# The look ahead mask is used to mask the future tokens in a
# sequence. In other words, the mask indicates which entries should
# not be used. This means that to predict the third word, only the
# first and second word will be used. Similarly to predict the
# fourth word, onl the first, second, and the third word will be
# used and so on.
def create_look_ahead_mask(size):
	mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
	return mask # (seq_len, seq_len)


# Scaled Dot Product Attention
# The attention function used by the transformer takes three inputs
# Q (query), K (key), V (value). The equation used to calculate the
# attention weights is:
# Attention(Q, K, V) = softmax_k ((QK^T)/sqrt(d_k)) * V.
# The dot-product attention is scaled by a factor of the square
# root of its depth. This is done because for large values of
# depth, the dot product grows large in magnitude pushing the
# softmax function where it has small gradients resulting in a very
# hard softmax.
# For example, consider that Q and K have a mean of 0 and variance
# of 1. Their matrix multiplication will have a mean of 0 and
# variance d_k. Hence, square root of d_k is used for scaling (and
# not any other number) because the matmul of Q and K should have a
# mean of 0 and variance of 1, and you get a gentler softmax.
# The mask is multiplied by -1e9 (close to negative infinity). This
# is done because the mask is summed with the scaled matrix
# multiplication of Q and K and is applied immediately before a
# softmax. The goal is to zero out these cells, and large negative
# inputs to softmax are near zero in the output.
def scaled_dot_product_attention(q, k, v, mask):
	# Calculate the attention weights.
	# q, k, v must have matching leading dimensions.
	# k, v must have matching penultimate dimensions, ie: 
	# seq_len_k = seq_len_v,
	# The mask has different shapes depending on its type(padding
	# or look ahead) but it mush be broadcastable for addition.

	# args:
	# q: query shape == (..., seq_len_q, depth)
	# k: key shape == (..., seq_len_k, depth)
	# v: value shape == (..., seq_len_v, depth_v)
	# mask: float tensor with shape broadcastable to
	#	(..., seq_len_q, seq_len_k). Defaults to None.
	# returns:
	# output, attention_weights
	matmul_qk = tf.matmul(q, k, transpose_b=True) # (..., seq_len_q, seq_len_k)

	# Scale matmul_qk.
	dk = tf.cast(tf.shape(k)[-1], tf.float32)
	scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

	# Add the mask to the scaled tensor.
	if mask is not None:
		scaled_attention_logits += (mask * -1e9)

	# Softmax is normalized on the last axis (seq_len_k) so that
	# the scores add up to 1.
	attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1) # (..., seq_len_q, seq_len_k)
	output = tf.matmul(attention_weights, v) # (..., seq_len_q, depth_v)
	return output, attention_weights


class TokenAndPositionEmbedding(layers.Layer):
	# Initialize the token and position embedding layer.
	# @param: context_size, int that details how the long the context
	#	(or sequence length) of an input sequence.
	# @param: vocab_size, int that describes how many "words" exist in
	#	the model's vocabulary.
	# @param: embedded_size, int that represents the size of the 
	#	embedding vector in this layer.
	# @return: returns nothing.
	def __init__(self, context_size, vocab_size, embedding_size, **kwargs):
		super(TokenAndPositionEmbedding, self).__init__()
		self.context_size = context_size
		self.vocab_size = vocab_size
		self.embedding_size = embedding_size

		# Word token embedding (or wte in GPT-2).
		self.token_embedding = layers.Embedding(input_dim=vocab_size,
												output_dim=embedding_size)

		# Word postitional encopding (or wpt in GPT-2).
		self.position_embedding = layers.Embedding(input_dim=context_size,
													output_dim=embedding_size)


	# Perform the following operations when this layer is called in a
	# model.
	# @param: inputs, an input tensor containing the input text.
	# @return: returns a tensor that is the sum of the position
	#	embedding and the token position.
	def call(self, inputs):
		max_len = tf.shape(inputs)[-1]
		positions = tf.range(start=0, limit=max_len, delta=1)
		positions = self.position_embedding(positions)
		x = self.token_embedding(inputs)
		return x + positions


	# Save any special configurations for this layer so that the model
	# can be saved and loaded without an issue. 
	# @param: takes no arguments.
	# @return: returns a copy of the config object for a tensorflow/
	#	keras layer.
	def get_config(self):
		# Needed for saving and loading model with custom Layer.
		config = super().get_config().copy()
		config.update({"context_size": self.context_size,
						"vocab_size": self.vocab_size,
						"embedding_size": self.embedding_size})
		return config


# Mask the upper half of the dot product matrix in self attention. This
# prevents the flow of information from future tokens to the current
# token. 1's in the lower triangle, counting from the lower right
# corner.
# @param: batch_size, an  int value setting the batch size (number of
#	samples per gradient update).
# @param: n_dest, an int that is the length of the the rows for the
#	attention matrix.
# @param: n_src, an int that is the number of columns for the attention
#	matrix.
# @param: dtype, a dtype to cast the matrix to (should be some sort of
#	bool in this function).
# @return: 
def causal_attn_mask(batch_size, n_dest, n_src, dtype):
	# Usually the n_dest and n_src are the same values (making the
	# attention matrix a square matrix). Start by initializing two
	# ranges of values (one for the n_dest and another for n_src). Then
	# compute the boolean matrix where only the "top" portion of the
	# matrix is False and all others are True.
	i = tf.range(n_dest)[:, None]
	j = tf.range(n_src)
	m = i >= j - n_src + n_dest

	# Convert the matrix to the dtype specified in the arguments (here,
	# it is expected to be tf.bool but it can also be some other
	# dtype).
	mask = tf.cast(m, dtype)

	# Reshape the matrix to be 3D (before matrix was 2D with dimensions
	# of n_dest x n_src. Now matrix is 1 x n_dest x n_src). Also create
	# another matrix of dimension 1 x 3 ([batch_size, 1, 1]).
	mask = tf.reshape(mask, [1, n_dest, n_src])
	mult = tf.concat(
		[tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.float32)], 0
	)

	# Return a tensor that will apply the causal attention mask to all
	# attention mask to all attention matrices in the batch.
	return tf.tile(mask, mult)


'''
class MultiHeadAttention(layers.Layer):
	def __init__(self, n_heads, key_dim, val_dim=None, dropout=0.0, use_bias=True, output_shape=None, attention_axes=None, kernel_initializer="glorot_uniform", bias_initializer="zeros", kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None, **kwargs):
		super(MultiHeadAttention, self).__init__()
		self.n_heads = n_heads
		self.key_dim = key_dim
		self.val_dim = val_dim if val_dim else key_dim
		self.dropout = dropout
		self.use_bias = self.use_bias
		self.output_shape = output_shape
		self.kernel_initializer = kernel_initializer
		self.bias_initializer = bias_initializer
		self.kernel_regularizer = kernel_regularizer
		self.bias_regularizer = bias_regularizer
		self.kernel_constraint = kernel_constraint
		self.bias_constraint = bias_constraint
		if attention_axes is not None and not isinstance(attention_axes, collections.abc.Sized):
			self.attention_axes = (attention_axes,)
		else:
			self.attention_axes = attention_axes
		self.built_from_signature = False

	def build_from_signature(self, query, value, key=None):
		self.built_from_signature = True
		if hasattr(query, "shape"):
			query_shape = te


	def call(self, inputs):
		pass


	# Save any special configurations for this layer so that the model
	# can be saved and loaded without an issue. 
	# @param: takes no arguments.
	# @return: returns a copy of the config object for a tensorflow/
	#	keras layer.
	def get_config(self):
		# Needed for saving and loading model with custom Layer.
		config = super().get_config().copy()
		config.update({"n_heads": self.n_heads,
						"embedding_size": self.embedding_size})
		return config
'''


'''
class MultiHeadAttention(layers.Layer):
	def __init__(self, n_heads, ff_dim, **kwargs):
		super(MultiHeadAttention, self).__init__()
		self.n_heads = n_heads
		self.ff_dim = ff_dim

		#assert ff_dim % n_heads == 0

		#self.depth = d_model // self.num_heads
		self.depth = ff_dim // n_heads

		self.wq = Dense(ff_dim)
		self.wk = Dense(ff_dim)
		self.wv = Dense(ff_dim)

		self.dense = Dense(ff_dim)


	def split_heads(self, x, batch_size):
			# Split the last dimension into (num_heads, depth).
			# Transpose the result such that the shape is (batch_size,
			# num_heads, seq_len, depth).
			#x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
			x = tf.reshape(x, (batch_size, -1, self.n_heads, self.depth))
			return tf.transpose(x, perm=[0, 2, 1, 3])


	def call(self, v, k, q):
			batch_size = tf.shape(q)[0]
			seq_len = tf.shape(q)[1]

			q = self.wq(q) # (batch_size, seq_len, d_model)
			k = self.wk(k) # (batch_size, seq_len, d_model)
			v = self.wv(v) # (batch_size, seq_len, d_model)

			q = self.split_heads(q, batch_size) # (batch_size, num_heads, seq_len_q, depth)
			k = self.split_heads(k, batch_size) # (batch_size, num_heads, seq_len_k, depth)
			v = self.split_heads(v, batch_size) # (batch_size, num_heads, seq_len_v, depth)

			# scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
			# attention_weights.shape == (batch_size, num_heads, seq_len_q, depth)
			#causal_mask = causal_attn_mask(batch_size, seq_len, seq_len, tf.float32)
			#scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)
			causal_mask = create_look_ahead_mask(seq_len)
			scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, causal_mask)
			scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads depth)
			concat_attention = tf.reshape(scaled_attention,
											(batch_size, -1, self.ff_dim))  # (batch_size, seq_len_q, d_model)
			output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)
			return output, attention_weights


	# Save any special configurations for this layer so that the model
	# can be saved and loaded without an issue. 
	# @param: takes no arguments.
	# @return: returns a copy of the config object for a tensorflow/
	#	keras layer.
	def get_config(self):
		# Needed for saving and loading model with custom Layer.
		config = super().get_config().copy()
		config.update({"n_heads": self.n_heads,
						#"embedding_size": self.embedding_size
						"ff_dim": self.ff_dim})
		return config
'''


class DecoderBlock(layers.Layer):
	# Initialize the GPT-2 decoder layer.
	# @param: n_heads, int setting the number of heads in the multihead
	#	attention layer.
	# @param: embedded_size, int that represents the size of the 
	#	embedding vector in this layer.
	# @param: ff_dim, an int (usually some power of 2) that is the
	#	number of neurons in the first layer of the feed forward neural
	#	network layer.
	# @param: rate, a float value between 0.0 and 1.0 that sets the
	#	dropout rate for the dropout layers after the multihead
	#	attention layer and feed forward neural network layer.
	# @return: returns nothing.
	def __init__(self, n_heads, embedding_size, ff_dim, rate, **kwargs):
		super(DecoderBlock, self).__init__()
		self.n_heads = n_heads
		self.embedding_size = embedding_size
		self.ff_dim = ff_dim
		self.rate = rate

		# Layers in the decoder block.
		#self.mha = Attention(causal=True)
		self.mha = MultiHeadAttention(num_heads=self.n_heads, head_size=self.ff_dim)
										#self.embedding_size)
		self.dropout_1 = Dropout(rate)
		self.layer_norm_1 = LayerNormalization(epsilon=1e-6)
		self.ffn = Sequential(
			[Dense(ff_dim, activation="relu"),
			Dense(self.embedding_size)])
		self.layer_norm_2 = LayerNormalization(epsilon=1e-6)
		self.dropout_2 = Dropout(rate)


	# Perform the following operations when this layer is called in a
	# model.
	# @param: inputs, an input tensor containing the input text.
	# @return: returns a tensor that is the result of passing the input
	#	through the multihead attention layer, first drouput layer,
	#	first normalization layer, feed forward neural network layer,
	#	second dropout layer, and finally the second normalization 
	#	layer.
	def call(self, inputs):
		# Extract the input_shape, batch_size, and seq_len from the
		# inputs shape.
		input_shape = tf.shape(inputs)
		batch_size = input_shape[0]
		seq_len = input_shape[1]

		# Pass the input through the layers. Note that for the
		# multihead attention layer, we need to add the attenion mask
		# to the top half of the attention matrix to prevent attention
		# to any "future" tokens.
		#causal_mask = causal_attn_mask(batch_size, seq_len, seq_len, tf.bool)
		#attention_output = self.mha(inputs, inputs, attention_mask=causal_mask)
		#attention_output, _ = self.mha(inputs, inputs, inputs)
		#attention_output = self.mha([inputs, inputs, inputs], mask=causal_mask)
		attention_output = self.dropout_1(attention_output)
		output_1 = self.layer_norm_1(inputs + attention_output)
		ffn_output = self.ffn(output_1)
		ffn_output = self.dropout_2(ffn_output)
		output_2 = self.layer_norm_2(output_1 + ffn_output)
		return output_2


	# Save any special configurations for this layer so that the model
	# can be saved and loaded without an issue. 
	# @param: takes no arguments.
	# @return: returns a copy of the config object for a tensorflow/
	#	keras layer.
	def get_config(self):
		# Needed for saving and loading model with custom Layer.
		config = super().get_config().copy()
		config.update({"n_heads": self.n_heads,
						"embedding_size": self.embedding_size,
						"ff_dim": self.ff_dim,
						"rate": self.rate})
		return config


class GPT2:
	# Initialize the GPT-2 class object.
	# @param: n_heads, int setting the number of heads in the multihead
	#	attention layer.
	# @param: n_layers, int value detailing how many decoder layers to
	#	include in the transformer model.
	# @param: vocab_size, int that describes how many "words" exist in
	#	the model's vocabulary.
	# @param: ff_dim, an int (usually some power of 2) that is the
	#	number of neurons in the first layer of the feed forward neural
	#	network layer.
	# @param: embedded_size, int that represents the size of the 
	#	embedding vector in this layer.
	# @param: context_size, int that details how the long the context
	#	(or sequence length) of an input sequence.
	# @param: dropout_rate, a float value between 0.0 and 1.0 that sets
	#	the dropout rate for the dropout layers after the multihead
	#	attention layer and feed forward neural network layer.
	# @param: optimizer, a string that details what optimizer should be
	#	used to train the model.
	# @param: loss, a string that is the loss function to use when
	#	training the model.
	# @param: metrics, a list containing the metrics to use when
	#	measuring model performance in training.
	# @return: returns nothing.
	def __init__(self, n_heads=12, n_layers=12, vocab_size=2**16, ff_dim=32,
				embedding_size=1024, context_size=1024, dropout_rate=0.1, 
				optimizer="adam", loss="sparse_categorical_crossentropy", 
				metrics=["accuracy"]):
		#super(GPT2, self).__init__()
		# Hyperparameters.
		self.n_heads = n_heads
		self.n_layers = n_layers
		self.vocab_size = vocab_size
		self.ff_dim = ff_dim
		self.embedding_size = embedding_size
		self.context_size = context_size
		self.dropout_rate = dropout_rate

		# Model compiler hyperparameters.
		self.optimizer = optimizer
		self.loss = loss
		self.metrics = metrics

		# Model architecture.
		self.input_layer = Input(shape=(self.context_size), dtype=tf.int32)
		self.embedding_layer = TokenAndPositionEmbedding(self.context_size, 
														self.vocab_size,
														self.embedding_size)
		self.decoder_layers = [DecoderBlock(self.n_heads, self.embedding_size, 
											self.ff_dim, self.dropout_rate) 
								for i in range(self.n_layers)]
		self.linear_layer = Dense(self.vocab_size)
		self.gpt_model = self.create_model()
		
		# Build and compile model. Print the model summary.
		self.gpt_model.compile(optimizer=self.optimizer, loss=self.loss, 
								metrics=self.metrics)
		print(self.gpt_model.summary())


	# Create the model (not meant to be called from outsite the GPT2
	# object).
	# @param: takes no arguments.
	# @return: returns a tensorflow/keras Model.
	def create_model(self):
		# Start with the input layer.
		inputs = self.input_layer

		# Pass all input through the embedding layer.
		x = self.embedding_layer(inputs)

		# Pass input through the decoder layer(s).
		for layer in self.decoder_layers:
			x = layer(x)

		# Pass input from the last decoder layer through to the linear
		# layer.
		output = self.linear_layer(x)
		
		# Return a tensorflow/keras model object.
		return Model(inputs=inputs, outputs=output)


	# Load an existing model and its hyperparameters.
	# @param: path_to_model_folder, the string path to a the model's
	#	save location.
	# @return: returns nothing.
	def load(self, path_to_model_folder):
		# Check for the existance of the path specified along with the
		# hyperparameters json file and the model's h5 model. Print an
		# error message and return the function if any of the files or
		# the folder don't exist.
		hparams_file = path_to_model_folder + "/hparams.json"
		h5_model_file = path_to_model_folder + "/gpt2_model.h5"
		if not os.path.exists(path_to_model_folder):
			print("Error: Path to folder does not exist.")
			return
		elif not os.path.exists(hparams_file):
			print("Error: Hyperparameter file in path to folder does not exist.")
			return
		elif not os.path.exists(h5_model_file):
			print("Error: Model h5 file in path to folder does not exist.")
			return

		# Load the hyperparameters and model from file.
		with open(hparams_file, "r") as json_file:
			hparams = json.load(json_file)
		self.n_heads = hparams["n_heads"]
		self.n_layers = hparams["n_layers"]
		self.vocab_size = hparams["vocab_size"]
		self.embedding_size = hparams["embedding_size"]
		self.ff_dim = hparams["ff_dim"]
		self.dropout_rate = hparams["dropout_rate"]
		self.optimizer = hparams["optimizer"]
		self.loss = hparams["loss"]
		self.metrics = hparams["metrics"]
		self.gpt_model = load_model(h5_model_file, 
									custom_objects={"TokenAndPositionEmbedding": TokenAndPositionEmbedding,
													"DecoderBlock": DecoderBlock})
		
		# Return the function.
		return


	# Save the model and its hyperparameters.
	# @param: path_to_model_folder, the string path to a the model's
	#	save location (will create folder and files if they does not
	#	already exist).
	# @return: returns nothing.
	def save(self, path_to_model_folder):
		# Check for the existance of the hyperparameters json file and
		# the model's h5 model.
		hparams_file = path_to_model_folder + "/hparams.json"
		h5_model_file = path_to_model_folder + "/gpt2_model.h5"
		if not os.path.exists(path_to_model_folder):
			#os.mkdir(path_to_model_folder)
			print("Error: Path to folder does not exist.")
			return

		# Save the hyperparameters and model to their respective files.
		hparams = {"n_heads": self.n_heads,	"n_layers": self.n_layers, 
					"vocab_size": self.vocab_size, "ff_dim": self.ff_dim,
					"embedding_size": self.embedding_size, 
					"dropout_rate": self.dropout_rate, 
					"optimizer": self.optimizer, "loss": self.loss,
					"metrics": self.metrics}
		with open(hparams_file, "w+") as json_file:
			json.dump(hparams, json_file, indent=4)
		self.gpt_model.save(h5_model_file)
		
		# Return the function.
		return


	# Train the model.
	# @param: x_train, a numpy array/tensorflow tensor/tensorflow data
	#	or dataset/generator or keras sequence as the input data.
	# @param: y_train, target data (that can also be a numpy array/
	#	tensorflow tensor/etc).
	# @param: validation, a tuple containing the data to evaluate loss
	#	and any model metrics at the end of the of an epoch.
	# @param: batch_size, an int value setting the batch size (number
	#	of samples per gradient update).
	# @param: epochs, an int value setting the number of epochs to
	#	train the model.
	# @param: verbose, an int value setting the verbosity mode (0, 1,
	#	2).
	# @param: callbacks, a list of all tensorflow/keras callback 
	#	objects.
	# @return: returns a History object from training the model.
	def train_model(self, x_train=None, y_train=None, validation=None, batch_size=32, epochs=1, verbose=1, callbacks=None):
		history = self.gpt_model.fit(x_train, y_train, validation, 
									batch_size=batch_size, epochs=epochs, 
									verbose=verbose, callbacks=callbacks)
		return history


	# Generate a text from the model given a prompt.
	# @param:
	# @param:
	# @param:
	# @param:
	# @param:
	# @return: returns nothing.
	def generate(self, input_prompt, max_length, top_k, num_return_sequences):
		pass


class TextGenerator(tf.keras.callbacks.Callback):
	# Initialize a tensorflow/keras callback for text generation.
	# @param: max_tokens, the maximum number of tokens to generate.
	# @param: start_tokens, the list of tokens that will prompt the
	#	model for text generation.
	# @param: index_to_word, the vocabulary for the model that 
	#	contains a dictionary mapping all index values to tokens.
	# @param: max_len, the maximum length of text to generate.
	# @param: top_k, the top_k results to sample from when generating
	#	the next token.
	# @param: print_every, the frequency (in epochs) to print a sample
	#	text generated from the model.
	# @return: returns nothing.
	def __init__(self, max_tokens, start_tokens, index_to_word, max_len, top_k=10, print_every=1):
		#super(TextGenerator, self).__init__():
		self.max_tokens = max_tokens
		self.start_tokens = start_tokens
		self.index_to_word = index_to_word
		self.max_len = max_len
		self.print_every = print_every
		self.k = top_k


	def sample_from(self, logits):
		logits, indices = tf.math.top_k(logits, k=self.k, sorted=True)
		indices = np.asarray(indices).astype("int32")
		preds = tf.keras.activations.softmax(tf.expand_dims(logits, 0))[0]
		preds = np.asarray(preds).astype("float32")
		return np.random.choice(indices, p=preds)


	def detokenize(self, number):
		return self.index_to_word[number]


	def on_epoch_end(self, epoch, logs=None):
		start_tokens = [_ for _ in self.start_tokens]
		if (epoch + 1) % self.print_every != 0:
			return
		num_tokens_generated = 0
		tokens_generated = []
		while num_tokens_generated <= self.max_tokens:
			pad_len = self.max_len - len(start_tokens)
			sample_index = len(start_tokens) - 1
			if pad_len < 0:
				x = start_tokens[:self.max_len]
				sample_index = max_len - 1
			elif pad_len > 0:
				x = start_tokens + [0] * pad_len
			else:
				x = start_tokens
			x = np.array([x])
			# y, _ = self.model.predict(x)
			y = self.model.predict(x)
			sample_token = self.sample_from(y[0][sample_index])
			tokens_generated.append(sample_token)
			start_tokens.append(sample_token)
			num_tokens_generated = len(tokens_generated)
		text = " ".join(
			[self.detokenize(_) for _ in self.start_tokens + tokens_generated]
		)
		print(f"Generated text:\n{text}\n")