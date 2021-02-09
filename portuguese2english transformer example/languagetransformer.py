# languagetransformer.py
# author: Diego Madgaleno
# Take the language transformer in the Tensorflow Portuguese to English
# example and convert it to a generic class object to apply to other
# similar applications.
# Sources:
# https://www.tensorflow.org/tutorials/text/transformer
# https://github.com/tensorflow/examples/blob/master/community/en/
# position_encoding.ipynb
# Python 3.7
# Tensorflow 2.4.0
# Windows/MacOS/Linux


import time
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


#'''
# Configuration code for allowing GPU usage on Tensorflow 2. Comment
# out when running on Tensorflow 1 on CPU.
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
session = tf.compat.v1.Session(config=config)
#'''


# Add positional encoding to give the model some information about
# the relative position of the words in the sentence. The
# positional encoding vector is added to the embedding vector.
# Embeddings represent a token in a d-dimensional space where
# tokens with similar meaning will be closer to each other. But the
# embeddings do not encode the relative position of words in a
# sentence. So after adding the positional encoding, words will be
# closer to each other based on similarity to their meaning and
# their position in the sentence, in the d-dimensional space.
# @param: pos, (numpy ndarray of ints) a 2D array with the number of 
#	positions in space.
# @param: i, (numpy ndarray of ints) a 2D array with the depth of the
#	vector.
# @param: d_model, (int) the dimensions of the output for the neural
#	network.
# @return: returns the product of the positions by the angle rate.
def get_angles(pos, i, d_model):
	# angle_rates = min_rate ** angle_rate_exponents
	# min_rate = 1/10000, angle_rate_exponents = (2 * (i // 2)) / 
	# np.float(d_model) OR np.linespace(0, 1, depth // 2) where depth =
	# d_model.
	angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
	return pos * angle_rates


# @param: position, (int) the position of a word in a sentence.
# @param: d_model, (int) the dimensions of the output for the
#	network.
# @return: returns the encoding for a given position (positional 
#	encoding) as a 32 bit float111.
def positional_encoding(position, d_model):
	angle_rads  = get_angles(np.arange(position)[:, np.newaxis],
							np.arange(d_model)[np.newaxis, :],
							d_model)

	# Apply sin to even indices in the array (2i).
	angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

	# Apply cos to odd indices in the array (2i + 1).
	angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

	pos_encoding = angle_rads[np.newaxis, ...]
	return tf.cast(pos_encoding, dtype=tf.float32)


# Mask all the pad tokens in the batch of sequence. It ensures that
# the model does not treat padding as the input. The mask indicates
# where pad value 0 is present: it outputs a 1 at those locations
# and a 0 otherwise.
# @param: seq, (tf.Tensor) a tensor representing a sequence
# @return: 
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
# @param: size, 
# @return:
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
# @param: q, (tensor)
# @param: k, ()
# @param: v, ()
# @param:
# @return:
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


# As the softmax normalization is done on K, its values decide the
# amount of importance given to Q. The output represents the
# multiplication of the attention weights and the V (value) vector.
# This ensures that the words you want to focus on are kept as-is
# and the irrelevant words are flushed out.


# Multihead Attention
# Mulit-head attention consists of four parts.
# -> Linear layers and split into heads
# -> Scaled dot-product attention
# -> Concatenation of heads
# -> Final linear layer
# Each multi-head attention block gets three inputs: Q (query),
# K (key), V (value). These are put through linear (Dense) layers
# and split up into multiple heads. The
# scaled_dot_product_attention defined above is applied to each
# head (broadcast for efficiency). An appropriate mask must be used
# in the attention step. The attention output for each head is then
# concatenated (using tf.transpose and tf.reshape) and put through
# a final Dense layer.
# Instead of one single attention head, Q, K, and V are split into
# multiple heads because it allows the model to jointly attend to 
# information at different positions from different
# representational spaces. After the split each head has a reduced
# dimensionality, so the total computation cost is the same as a
# single head attention with full dimensionality.
class MultiHeadAttention(tf.keras.layers.Layer):
	# Initialize the multiheadattention class object.
	# @param: d_model, (int) the dimensions of the output for the
	#	neural network.
	# @param: num_heads, (int) the number of heads in the 
	#	MultiHeadAttention layer.
	# @return: returns nothing.
	def __init__(self, d_model, num_heads):
		super(MultiHeadAttention, self).__init__()
		self.num_heads = num_heads
		self.d_model = d_model

		assert d_model % self.num_heads == 0

		self.depth = d_model // self.num_heads

		self.wq = tf.keras.layers.Dense(d_model)
		self.wk = tf.keras.layers.Dense(d_model)
		self.wv = tf.keras.layers.Dense(d_model)

		self.dense = tf.keras.layers.Dense(d_model)


	def split_heads(self, x, batch_size):
		# Split the last dimension into (num_heads, depth).
		# Transpose the result such that the shape is (batch_size,
		# num_heads, seq_len, depth).
		x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
		return tf.transpose(x, perm=[0, 2, 1, 3])


	def call(self, v, k, q, mask):
		batch_size = tf.shape(q)[0]

		q = self.wq(q) # (batch_size, seq_len, d_model)
		k = self.wk(k) # (batch_size, seq_len, d_model)
		v = self.wv(v) # (batch_size, seq_len, d_model)

		q = self.split_heads(q, batch_size) # (batch_size, num_heads, seq_len_q, depth)
		k = self.split_heads(k, batch_size) # (batch_size, num_heads, seq_len_k, depth)
		v = self.split_heads(v, batch_size) # (batch_size, num_heads, seq_len_v, depth)

		# scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
		# attention_weights.shape == (batch_size, num_heads, seq_len_q, depth)
		scaled_attention, attention_weights = scaled_dot_product_attention(q, k, 
																		v, mask)
		scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads depth)
		concat_attention = tf.reshape(scaled_attention,
										(batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)
		output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)
		return output, attention_weights


# At each location in the sequence, y, the MultiHeadAttention runs all
# attention heads heads across all other locations in the sequence,
# returning a new vector of the same length at each location.

# Point Wise Feed Forward Network
# Point wise feed forward network consists of two fully connected
# layers with a ReLU activation in between.
# Create and return a simple feed forward neural network.
# @param: d_model, (int) the dimensions of the output for the neural
#	network.
# @param: dff, (int) the number of units in the first layer of the
#	neural network.
# @return: returns a feed forward neural network consisting of two
#	fully connected layers.
def point_wise_feed_forward_network(d_model, dff):
	return tf.keras.Sequential([
			tf.keras.layers.Dense(dff, activation="relu"), # (batch_size, seq_len, dff)
			tf.keras.layers.Dense(d_model) #  (batch_size, seq_len, d_model)
	])


# Encoder and Decoder
# The transformer model follows the same general pattern as a
# standard sequence to sequence with attention model.
# -> The input sequence is passed through N encoder layers that
#	generates an output for each word/token in the sequence.
# -> The decoder attends on the encoders output and its own input
#	(self-attention) to predict the next word.

# Encoder Layer
# Each encoder layer consists of sublayers:
# -> Multi-head Attention (with padding mask)
# -> Point wise feed forward networks
# Each of these sublayers has a residual connection around it
# followed by a layer normalization. Residual connections help in
# avoiding the vanishing gradient problem in deep networks. The
# output of each layer is LayerNorm(x + Sublayer(x)). The
# normalization is done on the d_model (last) axis. There are N
# encoder layers in the transformer.
class EncoderLayer(tf.keras.layers.Layer):
	# Initialize the encoderlayer class object.
	# @param: d_model, (int) the dimensions of the output for the
	#	neural network.
	# @param: num_heads, (int) the number of heads in the 
	#	MultiHeadAttention layers.
	# @param: dff, (int) the number of units in the first layer of the
	#	neural network.
	# @param: rate, (float) the fraction of input units to drop in the
	#	dropout layers.
	# @return: returns nothing.
	def __init__(self, d_model, num_heads, dff, rate=0.1):
		super(EncoderLayer, self).__init__()

		self.mha = MultiHeadAttention(d_model, num_heads)
		self.ffn = point_wise_feed_forward_network(d_model, dff)

		self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
		self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

		self.dropout1 = tf.keras.layers.Dropout(rate)
		self.dropout2 = tf.keras.layers.Dropout(rate)


	def call(self, x, training, mask):
		attn_output, _ = self.mha(x, x, x, mask) # (batch_size, input_seq_len, d_model)
		attn_output = self.dropout1(attn_output, training=training)
		out1 = self.layernorm1(x + attn_output) # (batch_size, input_seq_len, d_model)

		ffn_output = self.ffn(out1) # (batch_size, input_seq_len, d_model)
		ffn_output = self.dropout2(ffn_output, training=training)
		out2 = self.layernorm2(out1 + ffn_output) # (batch_size, input_seq_len, d_model)

		return out2


# Decoder Layer
# The decoder layer consists of sublayers:
# -> Masked multi-head attention (with look ahead mask and padding
#	mask)
# -> Multi-head Attention (with padding mask). V (value) and K
#	(key) recieve the encoder output as inputs. Q (query) receives
#	the output from the masked multi-head attention sublayer.
# -> Point wise feed forward networks
# Each of these sublayers has a residual connection around it
# followed by a layer normalization. The output of each sublayer
# is LayerNorm(x + Sublayer(x)). The normalization is done on the
# d_model (last) axis.
# There are N decoder layers in the transformer. As Q recieves the
# output from decoder's first attention block, and K recieves the
# encoder output, the attention weights represent the importance
# given to the decoder's input based on the encoder's output. In
# other words, the decoder predicts the next word by looking at the
# encoder output and self-attending to its own output. (See the 
# demonstration above in the scaled dot product attention section).
class DecoderLayer(tf.keras.layers.Layer):
	# Initialize the decoderlayer class object.
	# @param: d_model, (int) the dimensions of the output for the
	#	neural network.
	# @param: num_heads, (int) the number of heads in the 
	#	MultiHeadAttention layers.
	# @param: dff, (int) the number of units in the first layer of the
	#	neural network.
	# @param: rate, (float) the fraction of input units to drop in the
	#	dropout layers.
	# @return: returns nothing.
	def __init__(self, d_model, num_heads, dff, rate=0.1):
		# Allow the decoderlayer class to manage inheritance from
		# tf.keras.layers.Layer class.
		super(DecoderLayer, self).__init__()

		# Intialize the two MultiHeadAttention objects.
		self.mha1 = MultiHeadAttention(d_model, num_heads)
		self.mha2 = MultiHeadAttention(d_model, num_heads)

		# Intialize the feed forward network to follow the
		# MultiHeadAttention objects.
		self.ffn = point_wise_feed_forward_network(d_model, dff)

		# Intialize the layer normalization layers that follow each
		# MultiHeadAttention and the feed forward network objects.
		self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
		self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
		self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

		# Intialize the dropout layers that follow each
		# MultiHeadAttention layer and the feed forward network objects
		# but comes before the layer normalization layers.
		self.dropout1 = tf.keras.layers.Dropout(rate)
		self.dropout2 = tf.keras.layers.Dropout(rate)
		self.dropout3 = tf.keras.layers.Dropout(rate)


	def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
		# enc_output.shape == (batch_size, input_seq_len, d_model)
		attn1, attention_weights_block1 = self.mha1(x, x, x, look_ahead_mask) # (batch_size, target_seq_len, d_model)
		attn1 = self.dropout1(attn1, training=training)
		out1 = self.layernorm1(attn1 + x)

		attn2, attention_weights_block2 = self.mha2(enc_output, enc_output, out1,
													padding_mask) # (batch_size, target_seq_len, d_model)
		attn2 = self.dropout2(attn2, training=training)
		out2 = self.layernorm2(attn2 + out1) # (batch_size, target_seq_len, d_model)

		ffn_output = self.ffn(out2) # (batch_size, target_seq_len, d_model)
		ffn_output = self.dropout3(ffn_output, training=training)
		out3 = self.layernorm3(ffn_output + out2) # (batch_size, target_seq_len, d_model)
		
		return out3, attention_weights_block1, attention_weights_block2\


# Encoder
# The Encoder consists of:
# -> Input Embedding 
# -> Positional Encoding
# -> N encoder layers
# The input is put through an embedding which is summed with the
# positional encoding. The output of this summation is then input
# to the encoder layers. The output of the encoder is the input to
# the decoder.
class Encoder(tf.keras.layers.Layer):
	# Initialize the encoder class object.
	# @param: num_layers, (int) the number of layers in the Encoder and
	#	Decoder.
	# @param: d_model, (int) the dimensions of the output for the
	#	neural network.
	# @param: num_heads, (int) the number of heads in the 
	#	MultiHeadAttention layers.
	# @param: dff, (int) the number of units in the first layer of the
	#	neural network.
	# @param: input_vocab_size, (int) the size of the vocabulary for
	#	input text/language.
	# @param: maximum_position_encoding, (int) the highest positional
	#	encoding value for the input.
	# @param: rate, (float) the fraction of input units to drop in the
	#	dropout layers.
	# @return: returns nothing.
	def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, 
				maximum_position_encoding, rate=0.1):
		# Allow the encoder class to manage inheritance from
		# tf.keras.layers.Layer class.
		super(Encoder, self).__init__()

		# Save the model dimensions and number of (decoder) layers for
		# the decoder.
		self.d_model = d_model
		self.num_layers = num_layers

		# Initialize the embedding layer for the encoder and the
		# positional encodings for the maximum positional encoding.
		self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
		self.pos_encoding = positional_encoding(maximum_position_encoding, 
												self.d_model)

		# Initialize the encoder layers.
		self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate)
							for _ in range(num_layers)]

		# Initialize the dropout layer for after the encoder layers.
		self.dropout = tf.keras.layers.Dropout(rate)


	def call(self, x, training, mask):
		seq_len = tf.shape(x)[1]

		# Adding embedding and position encoding.
		x = self.embedding(x) # (batch_size, input_seq_len, d_model)
		x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
		x += self.pos_encoding[:, :seq_len, :]

		x = self.dropout(x, training=training)

		for i in range(self.num_layers):
			x = self.enc_layers[i](x, training, mask)

		return x # (batch_size, input_seq_len, d_model)


# Decoder
# The Decoder consists of:
# -> Output Embedding
# -> Positional Embedding
# -> N decoder layers
# The target is put through an embedding which is summed with the
# positional encoding. The output of this summation is the input to
# the decoder layers. The output of the decoder is the input to the
# final linear layer.
class Decoder(tf.keras.layers.Layer):
	# Initialize the decoder class object.
	# @param: num_layers, (int) the number of layers in the Encoder and
	#	Decoder.
	# @param: d_model, (int) the dimensions of the output for the
	#	neural network.
	# @param: num_heads, (int) the number of heads in the 
	#	MultiHeadAttention layers.
	# @param: dff, (int) the number of units in the first layer of the
	#	neural network.
	# @param: target_vocab_size, (int) the size of the vocabulary for
	#	output text/language.
	# @param: maximum_position_encoding, (int) the highest positional
	#	encoding value for the output.
	# @param: rate, (float) the fraction of input units to drop in the
	#	dropout layers.
	# @return: returns nothing.
	def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size, 
				maximum_position_encoding, rate=0.1):
		# Allow the decoder class to manage inheritance from
		# tf.keras.layers.Layer class.
		super(Decoder, self).__init__()

		# Save the model dimensions and number of (decoder) layers for
		# the decoder.
		self.d_model = d_model
		self.num_layers = num_layers

		# Initialize the embedding layer for the decoder and the
		# positional encodings for the maximum positional encoding.
		self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
		self.pos_encoding = positional_encoding(maximum_position_encoding, 
												d_model)

		# Initialize the decoder layers.
		self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate)
							for _ in range(num_layers)]

		# Initialize the dropout layer for after the decoder layers.
		self.dropout = tf.keras.layers.Dropout(rate)


	def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
		seq_len = tf.shape(x)[1]
		attention_weights = {}

		x = self.embedding(x) # (batch_size, target_seq_len, d_model)
		x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
		x += self.pos_encoding[:, :seq_len, :]

		x = self.dropout(x, training=training)

		for i in range(self.num_layers):
			x, block1, block2 = self.dec_layers[i](x, enc_output, training, 
													look_ahead_mask, padding_mask)
			attention_weights["decoder_layer{}_block1".format(i + 1)] = block1
			attention_weights["decoder_layer{}_block2".format(i + 1)] = block2

		# x.shape == (batch_size, target_seq_len, d_model)
		return x, attention_weights



# The transformer consists of the encoder, decoder, and a final
# linear layer. The output of the decoder is the input to the
# linear layer and its output is returned.
class Transformer(tf.keras.Model):
	# Intialize the transformer class object.
	# @param: num_layers, (int) the number of layers in the Encoder and
	#	Decoder.
	# @param: d_model, (int) the dimensions of the output for the
	#	neural network.
	# @param: num_heads, (int) the number of heads in the 
	#	MultiHeadAttention layers.
	# @param: dff, (int) the number of units in the first layer of the
	#	neural network.
	# @param: input_vocab_size, (int) the size of the vocabulary for
	#	input text/language.
	# @param: target_vocab_size, (int) the size of the vocabulary for
	#	output text/language.
	# @param: pe_input, (int) the highest positional incoding value for
	#	the input.
	# @param: pe_target, (int) the highest positional encoding value 
	#	for the output.
	# @param: rate, (float) the fraction of input units to drop in the
	#	dropout layers.
	# @return: returns nothing.
	def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, 
				target_vocab_size, pe_input, pe_target, rate=0.1):
		# Allow the transformer class to manage inheritance from the
		# tf.keras.Model class.
		super(Transformer, self).__init__()

		# Initialize the encoder for the transformer.
		self.encoder = Encoder(num_layers, d_model, num_heads, dff,
								input_vocab_size, pe_input, rate)
		
		# Initialize the decoder for the transformer.
		self.decoder = Decoder(num_layers, d_model, num_heads, dff, 
								target_vocab_size, pe_target, rate)

		# Initialize the final Dense layer for the transformer after
		# the decoder.
		self.final_layer = tf.keras.layers.Dense(target_vocab_size)


	# Run a call to the transformer itself. Pass in an input and expect
	# an outut. If the training argument is True, this will train the
	# layers of the transformer.
	# @param: inp,
	# @param: tar,
	# @param: training, (bool)
	# @param: enc_padding_mask,
	# @param: look_ahead_mask,
	# @param: dec_padding_mask,
	# @return: returns
	def call(self, inp, tar, training, enc_padding_mask, look_ahead_mask, 
			dec_padding_mask):
		enc_output = self.encoder(inp, training, enc_padding_mask) # (batch_size, inp_seq_len, d_model)

		# dec_output.shape == (batch_size, tar_seq_len, d_model)
		dec_output, attention_weights = self.decoder(tar, enc_output, training,
													look_ahead_mask, dec_padding_mask)

		final_output = self.final_layer(dec_output) # (batch_size, tar_seq_len, target_vocab_size)
		return final_output, attention_weights


class LanguageTransformer():
	# Initialize the languagetransformer class object.
	# @param: num_layers, (int) the number of layers in the Encoder and
	#	Decoder.
	# @param: d_model, (int) the dimensions of the output for the
	#	neural network.
	# @param: num_heads, (int) the number of heads in the 
	#	MultiHeadAttention layers.
	# @param: dff, (int) the number of units in the first layer of the
	#	neural network.
	# @param: input_vocab_size, (int) the size of the vocabulary for
	#	input text/language.
	# @param: target_vocab_size, (int) the size of the vocabulary for
	#	output text/language.
	# @param: dropout_rate, (float) the fraction of input units to drop
	#	in the dropout layers.
	# @param: tokenizer_in, (tensorflow_datasets.deprecated.text 
	#	SubwordTextEncoder, TextEncoder, ByteTextEncoder, or 
	#	TokenTextEncoder object) an abstract class for converting 
	#	between text and integers. This is the text encoder/tokenizer
	#	for the input text/language.
	# @param: tokenizer_out, (tensorflow_datasets.deprecated.text 
	#	SubwordTextEncoder, TextEncoder, ByteTextEncoder, or 
	#	TokenTextEncoder object) an abstract class for converting 
	#	between text and integers. This is the text encoder/tokenizer
	#	for the output text/language.
	# @return: returns nothing.
	def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, 
				target_vocab_size, dropout_rate, tokenizer_in, tokenizer_out):
		# Initialize a transformer with the following arguments.
		self.transformer = Transformer(num_layers, d_model, num_heads, dff,
								input_vocab_size, target_vocab_size,
								pe_input=input_vocab_size,
								pe_target=target_vocab_size,
								rate=dropout_rate)

		# Initialize the learning rate and optimizer for the model.
		self.learning_rate = CustomSchedule(d_model)
		self.optimizer = tf.keras.optimizers.Adam(self.learning_rate, 
													beta_1=0.9, beta_2=0.98,
													epsilon=1e-9)

		# Initialize a loss object that computes the crossentropy loss
		# between the labels and predictions. In particular, the
		# predicted output (y_pred) is expected to be a logits tensor
		# and there is no type of loss reduction.
		self.loss_object = tf.keras.losses.\
							SparseCategoricalCrossentropy(from_logits=True,
															reduction="none")

		# Initialize the training loss and accuracy variables. These 
		# compute the weighted mean of the given values.
		self.train_loss = tf.keras.metrics.Mean(name="train_loss")
		self.train_accuracy = tf.keras.metrics.Mean(name="train_accuracy")

		# Store all remaining arguments to the class.
		self.tokenizer_in = tokenizer_in
		self.tokenizer_out = tokenizer_out


	# Train the transformer.
	# @param: train_dataset, (tensorflow.data.dataset) dataset 
	#	containing the input and output text samples in their own 
	#	respective tensors.
	# @param: epochs, (int) the number of epochs to train the 
	#	transformer.
	# @param: checkpoint_path, (str) the string of the path to the 
	#	checkpoints.
	# @return: returns nothing.
	def fit(self, train_dataset, epochs, checkpoint_path="./checkpoints/train"):
		# Create the checkpoint path and the checkpoint manager. This will
		# be used to save checkpoints every n epochs.
		# checkpoint_path = "./checkpoints/train"
		ckpt = tf.train.Checkpoint(transformer=self.transformer, 
									optimizer=self.optimizer)
		ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path)
		# ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, 
		# 											max_to_keep=5)

		# If a checkpoint exists, restore the latest checkpoint.
		if ckpt_manager.latest_checkpoint:
			ckpt.restore(ckpt_manager.latest_checkpoint)
			print("Latest checkpoint restored!")

		# The @tf.function trace-compiles train_step into a TF graph for
		# faster execution. The function specializes to the precise shape
		# of the argument tensors. To avoid re-tracing due to the variable
		# sequence lengths or variable batch sizes (the last batch is
		# smaller), use input_signature to specify more generic shapes.
		train_step_signature = [
			tf.TensorSpec(shape=(None, None), dtype=tf.int64),
			tf.TensorSpec(shape=(None, None), dtype=tf.int64),
		]

		# Training step in training the transformer.
		# @param: inp, (tensorflow.python.framework.ops.EagerTensor of
		#	tf.int32) a sample of text from the input/language. The 
		#	tensor is (batch_size, seq_len) shaped. Sample (sentenes) 
		#	that are shorter than the seq_len are padded to fill the
		#	shape. Each entry (in the sentence) is a tokenized/encoded
		#	word from the input (language).
		# @param: tar, (tensorflow.python.framework.ops.EagerTensor of
		#	tf.int32) a sample of text from the output/language. The 
		#	tensor is (batch_size, seq_len) shaped. Sample (sentenes) 
		#	that are shorter than the seq_len are padded to fill the
		#	shape. Each entry (in the sentence) is a tokenized/encoded
		#	word from the output (language).
		# @return: returns nothing.
		@tf.function(input_signature=train_step_signature)
		def train_step(inp, tar):
			tar_inp = tar[:, :-1]
			tar_real = tar[:, 1:]

			enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

			with tf.GradientTape() as tape:
				predictions, _ = self.transformer(inp, tar_inp, True, enc_padding_mask,
												combined_mask, dec_padding_mask)
				loss = self.loss_function(tar_real, predictions)

			gradients = tape.gradient(loss, self.transformer.trainable_variables)
			optimizer.apply_gradients(zip(gradients, self.transformer.trainable_variables))

			self.train_loss(loss)
			self.train_accuracy(self.accuracy_function(tar_real, predictions))

		# Portugese is used as the input language and English is the target
		# language.
		for epoch in range(epochs):
			start = time.time()

			self.train_loss.reset_states()
			self.train_accuracy.reset_states()

			# inp -> portugese, tar -> english.
			for (batch, (inp, tar)) in enumerate(train_dataset):
				self.train_step(inp, tar)

				if batch % 50 == 0: 
					print("Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}".format(
							epoch + 1, batch, self.train_loss.result(), 
							self.train_accuracy.result()))

			if (epoch + 1) % 5 == 0:
				ckpt_save_path = ckpt_manager.save()
				print("Saving checkpoint for epoch {} at {}".format(epoch + 1, 
						ckpt_save_path))

			print("Epoch {} Loss {:.4f} Accuracy {:.4f}".format(epoch + 1,
					self.train_loss.result(), self.train_accuracy.result()))
			print("Time take for 1 epoch: {} secs\n".format(time.time() - start))


	# Save the transformer.
	# @param: save_path, (str) the string of the path to the saved
	#	transformer model.
	# @return: returns nothing.
	def save_transformer_model(self, save_path):
		self.transformer.save(save_path)


	# Load a transformer model.
	# @param: save_path, (str) the string of the path to the saved
	#	transformer model.
	# @return: returns nothing.
	def load_transformer_model(self, save_path):
		self.transformer.load_model(save_path)


	# Evaluate
	# The following steps are used for evaluation:
	# -> Encode the input sequence using the Portugese tokenizer
	#	(tokenizer_pt). Moreover, add the start and end token so the
	#	input is equivalent to what the model is trained with. This is
	#	the encoder input.
	# -> The decoder input is the start token == tokenizer_en.vocab_size.
	# -> Calculate the padding masks and the look ahead masks.
	# -> The decoder then outputs the predictions by looking at the
	#	encoder output and its own output (self-attention).
	# -> Select the last word and calculate the argmax of that.
	# -> Concatenate the predicted word to the decoder input as pass it
	#	to the decoder.
	# -> In this approach, the decoder predicts the next word based on
	#	the previous words it predicted.
	# Note the model used here has less capacity to keep the example
	# relatively faster so the predictions maybe less right. To
	# reproduce the results in the paper, use the entire dataset and
	# base transformer model or transformer XL, by changing the
	# hyperparameters above.

	# Evaluate the transformer's ability to predict the next word in
	# the sentence (can be for translation or text generation).
	# @param: inp_sentence,
	# @return:
	def evaluate(self, inp_sentence):
		# start_token = [tokenizer_pt.vocab_size]
		# end_token = [tokenizer_pt.vocab_size + 1]
		start_token = [self.tokenizer_in.vocab_size]
		end_token = [self.tokenizer_in.vocab_size + 1]

		# inp sentence is portuguese, hence adding the start and end
		# token.
		# inp_sentence = start_token + tokenizer_pt.encode(inp_sentence) + end_token
		inp_sentence = start_token + self.tokenizer_in.encode(inp_sentence) + end_token
		encoder_input = tf.expand_dims(inp_sentence, 0)

		# as the target is in english, the first word to the
		# transformer should be the english start token.
		# decoder_input = [tokenizer_en.vocab_size]
		decoder_input = [self.tokenizer_out.vocab_size]
		output = tf.expand_dims(decoder_input, 0)

		for i in range(MAX_LENGTH):
			enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
				encoder_input, output)

			# predictions.shape == (batch_size, seq_len, vocab_size)
			predictions, attention_weights = self.transformer(encoder_input, 
																output,	False, 
																enc_padding_mask,
																combined_mask,
																dec_padding_mask)

			# select the last word from the seq_len dimension
			predictions = predictions[:, -1:, :] # (batch_size, 1, vocab_size)

			predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

			# return the result if the prediction_id is equal to the end token.
			if predicted_id == tokenizer_en.vocab_size + 1:
				return tf.squeeze(output, axis=0), attention_weights

			# concatenate the predicted_id to the output which is given
			# to the decoder as its input.
			output = tf.concat([output, predicted_id], axis=-1)

		return tf.squeeze(output, axis=0), attention_weights



	# Calculate the loss of the model and return it.
	# @param: real,
	# @param: pred,
	# @return:
	def loss_function(self, real, pred):
		mask = tf.math.logical_not(tf.math.equal(real, 0))
		loss_ = loss_object(real, pred)

		mask = tf.cast(mask, dtype=loss_.dtype)
		loss_ *= mask

		return tf.reduce_sum(loss_) / tf.reduce_sum(mask)


	# Calculate the accuracy of the model and return it.
	# @param: real,
	# @param: pred,
	# @return:
	def accuracy_function(self, real, pred):
		accuracies = tf.equal(real, tf.argmax(pred, axis=2))

		mask = tf.math.logical_not(tf.math.equal(real, 0))
		accuracies = tf.math.logical_and(mask, accuracies)

		accuracies = tf.cast(accuracies, dtype=tf.float32)
		mask = tf.cast(mask, dtype=tf.float32)

		return tf.reduce_sum(accuracies) / tf.reduce_sum(mask)


	# Plot the attention weights.
	# @param: attention,
	# @param: sentence,
	# @param: result,
	# @param: layer,
	# @return: returns nothing.
	def plot_attention_weights(self, attention, sentence, result, layer):
		fig = plt.figure(figsize=(16, 8))

		# sentence = tokenizer_pt.encode(sentence)
		sentence = self.tokenizer_in.encode(sentence)

		attention = tf.squeeze(attention[layer], axis=0)

		for head in range(attention.shape[0]):
			ax = fig.add_subplot(2, 4, head + 1)

			# plot the attention weights.
			ax.matshow(attention[head][:-1, :], cmap="viridis")

			fontdict = {"fontsize": 10}

			ax.set_xticks(range(len(sentence) + 2))
			ax.set_yticks(range(len(result)))

			ax.set_ylim(len(result) - 1.5, -0.5)

			# ax.set_xticklabels(
			# 	["<start>"] + [tokenizer_pt.decode([i]) for i in sentence] + ["<end>"])
			ax.set_xticklabels(
				["<start>"] + [self.tokenizer_in.decode([i]) for i in sentence] + ["<end>"])

			# ax.set_yticklabels([tokenizer_en.decode([i]) for i in result
			# 					if i < tokenizer_en.vocab_size],
			# 					fontdict=fontdict)
			ax.set_yticklabels([self.tokenizer_out.decode([i]) for i in result
								if i < self.tokenizer_out.vocab_size],
								fontdict=fontdict)

			ax.set_xlabel("Head {}".format(head + 1))

		plt.tight_layout()
		plt.show()


	'''
	def translate(sentence, plot=""):
		result, attention_weights = evaluate(sentence)

		predicted_sentence = tokenizer_en.decode([i for i in result
												if i < tokenizer_en.vocab_size])

		print("Input: {}".format(sentence))
		print("Predicted translation: {}".format(predicted_sentence))

		if plot:
			plot_attention_weights(attention_weights, sentence, result, plot)
	'''


	# Predict the output given an input sentence and return it.
	# @param: sentence, ()
	# @param: plot, (str) 
	# @return: Return a (decoded) output sentence.
	def predict(self, sentence, plot=""):
		result, attention_weights = self.evaluate(sentence)

		# predicted_sentence = tokenizer_en.decode([i for i in result
		# 										if i < tokenizer_en.vocab_size])
		predicted_sentence = self.tokenizer_out.decode([i for i in result
												if i < self.tokenizer_out.vocab_size])

		print("Input: {}".format(sentence))
		print("Predicted output: {}".format(predicted_sentence))

		if plot:
			self.plot_attention_weights(attention_weights, sentence, result, plot)

		return predicted_sentence
