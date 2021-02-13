# textclassifiertransformer.py
# Implement a transformer block as a keras layer and use it for text
# classification.
# source: https://keras.io/examples/nlp/text_classification_with_
# transformer/
# Python 3.7
# Tensorflow 1.14/1.15/2.4.0
# Windows/MacOS/Linux


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


#'''
# Configuration code for allowing GPU usage on Tensorflow 2. Comment
# out when running on Tensorflow 1 on CPU.
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
session = tf.compat.v1.Session(config=config)
#'''


class TransformerBlock(layers.Layer):
	def __init__(self, embedded_dim, num_heads, ff_dim, rate=0.1):
		super(TransformerBlock, self).__init__()
		self.attention = layers.MultiHeadAttention(num_heads=num_heads, 
													key_dim=embedded_dim)
		self.ffn = keras.Sequential(
			[layers.Dense(ff_dim, activation="relu"),
			layers.Dense(embedded_dim)])
		self.layer_norm_1 = layers.LayerNormalization(epsilon=1e-6)
		self.layer_norm_2 = layers.LayerNormalization(epsilon=1e-6)
		self.dropout_1 = layers.Dropout(rate)
		self.dropout_2 = layers.Dropout(rate)


	def call(self, inputs, training):
		attention_output = self.attention(inputs, inputs)
		attention_output = self.dropout_1(attention_output, training=training)
		output_1 = self.layer_norm_1(inputs + attention_output)
		ffn_output = self.ffn(output_1)
		ffn_output = self.dropout_2(ffn_output, training=training)
		return self.layer_norm_2(output_1 + ffn_output)


class TokenAndPositionEmbedding(layers.Layer):
	def __init__(self, max_len, vocab_size, embedded_dim):
		super(TokenAndPositionEmbedding, self).__init__()
		self.token_embedding = layers.Embedding(input_dim=vocab_size,
												output_dim=embedded_dim)
		self.position_embedding = layers.Embedding(input_dim=max_len,
													output_dim=embedded_dim)


	def call(self, x):
		max_len = tf.shape(x)[-1]
		positions = tf.range(start=0, limit=max_len, delta=1)
		positions = self.position_embedding(positions)
		x = self.token_embedding(x)
		return x + positions


def main():
	# Download and prepare dataset.
	vocab_size = 20000 # Only consider the top 20K samples.
	max_len = 200 # Only consider the first 200 words of each movie review
	(x_train, y_train), (x_val, y_val) = keras.datasets.imdb.load_data(num_words=vocab_size)
	print(len(x_train), "Training sequences")
	print(len(x_val), "Validation sequences")
	x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_len)
	x_val = keras.preprocessing.sequence.pad_sequences(x_val, maxlen=max_len)

	# Create the classifier model using the transformer layer.
	# Transformer layer outputs one vector for each time step of our
	# input sequence. Here, we take the mean across all time steps and
	# use a feed forward network on top to classify text.
	embedded_dim = 32 # Embedding size for each token
	num_heads = 2 # Number of attention heads
	ff_dim = 32 # Hidden layer size in feed forward network inside transformer

	inputs = layers.Input(shape=(max_len,))
	embedding_layer = TokenAndPositionEmbedding(max_len, vocab_size, embedded_dim)
	x = embedding_layer(inputs)
	transformer_block = TransformerBlock(embedded_dim, num_heads, ff_dim)
	x = transformer_block(x)
	x = layers.GlobalAveragePooling1D()(x)
	x = layers.Dropout(0.1)(x)
	x = layers.Dense(20, activation="relu")(x)
	x = layers.Dropout(0.1)(x)
	outputs = layers.Dense(2, activation="softmax")(x)

	model = keras.Model(inputs=inputs, outputs=outputs)

	# Train and evaluate.
	model.compile(optimizer="adam",
					loss="sparse_categorical_crossentropy",
					metrics=["accuracy"])
	print(model.summary())
	history = model.fit(x_train, y_train, batch_size=32, epochs=2,
						validation_data=(x_val, y_val))


if __name__ == '__main__':
	main()