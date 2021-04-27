# bert.py
# author: Diego Magdaleno
# Take the BERT model implemented in mlm_bert.py (see Keras_Examples
# repository, specifically the Masked_Language_Modeling_BERT folder)
# and implement it here as a class instead of a function.
# Tensorflow 1.14/1.15/2.4
# Python 3.7
# Windows/MacOS/Linux


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


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


class EncoderBlock(layers.Layer):
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
		super(EncoderBlock, self).__init__()
		self.mha = layers.MultiHeadAttention(n_heads, embedding_size // n_heads)
		self.dropout_1 = layers.Dropout(rate)
		self.layer_norm_1 = layers.LayerNormalization(epsilon=1e-6)
		self.ffn = keras.Sequential(
			[
				layers.Dense(ff_dim, activation="relu"),
				layers.Dense(embedding_size),
			]
		)
		self.dropout_2 = layers.Dropout(rate)
		self.layer_norm_2 = layers.LayerNormalization(epsilon=1e-6)


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
		attention_output = self.mha(inputs, inputs, inputs)
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


class MaskedLanguageModel(tf.keras.Model):
	def train_step(self, inputs):
		if len(inputs) == 3:
			features, labels, sample_weight = inputs
		else:
			features, labels = inputs
			sample_weight = None

		with tf.GradientTape() as tape:
			predictions = self(features, training=True)
			loss = loss_fn(labels, predictions, sample_weight=sample_weight)

		# Compute gradients.
		trainable_vars = self.trainable_variables
		gradients = tape.gradient(loss, trainable_vars)

		# Update weights.
		self.optimizer.apply_gradients(zip(gradients, trainable_vars))

		# Compute our own metrics.
		loss_tracker.update_state(loss, sample_weight=sample_weight)

		# Return a dict mapping metric names to current value.
		return {"loss": loss_tracker.result()}


	@property
	def metrics(self):
		# List the "metric" objects here so that reset_states() can be
		# called automatically at the start of each epoch or at the
		# start of evaluate(). If the property is not implemented,
		# reset_states() will have to be called at a time of choosing.
		return [loss_tracker]


class BERT:
	# Initialize the BERT class object.
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
				metrics=["accuracy"], model_name=None):
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
		self.input_layer = Input(shape=(self.context_size), dtype=tf.int64)
		self.embedding_layer = TokenAndPositionEmbedding(
			self.context_size, self.vocab_size, self.embedding_size
		)
		self.encoder_layers = [EncoderBlock(
				self.n_heads, self.embedding_size, self.ff_dim, self.dropout_rate
			) 
			for i in range(self.n_layers)]
		self.linear_layer = Dense(
			self.vocab_size, name="mlm_cls", activation="softmax"
		)
		self.bert_mlm_model = self.create_model(model_name)
		
		# Build and compile model. Print the model summary.
		self.bert_mlm_model.compile(
			optimizer=self.optimizer, loss=self.loss, metrics=self.metrics
		)
		print(self.bert_mlm_model.summary())


	# Create the model (not meant to be called from outsite the BERT
	# object).
	# @param: model_name, a string that names the model.
	# @return: returns a tensorflow/keras Model.
	def create_mlm_bert(self, model_name):
		if model_name is None:
			model_name = "masked_bert_model"

		# Start with the input layer.
		inputs = self.input_layer

		# Pass all input through the embedding layer.
		x = self.embedding_layer(inputs)

		# Pass input through the decoder layer(s).
		for layer in self.encoder_layers:
			x = layer(x)

		# Pass input from the last decoder layer through to the linear
		# layer.
		mlm_output = self.linear_layer(x)

		mlm_model = MaskedLanguageModel(inputs, mlm_output, name=model_name)
		return mlm_model

		'''
		word_embeddings = layers.Embedding(
			config.VOCAB_SIZE, config.EMBED_DIM, name="word_embeddings"
		)(inputs)
		position_embeddings = layers.Embedding(
			input_dim=config.MAX_LEN,
			output_dim=config.EMBED_DIM,
			weights=[get_pos_encoding_matrix(config.MAX_LEN, config.EMBED_DIM)],
			name="position_embeddings",
		)(tf.range(start=0, limit=config.MAX_LEN, delta=1))
		embeddings = word_embeddings + position_embeddings

		encoder_output = embeddings
		for i in range(config.NUM_LAYERS):
			encoder_output = bert_module(encoder_output, encoder_output, encoder_output, i, config)

		mlm_output = layers.Dense(config.VOCAB_SIZE, name="mlm_cls", activation="softmax")(
			encoder_output
		)
		mlm_model = MaskedLanguageModel(inputs, mlm_output, name="masked_bert_model")

		optimizer = keras.optimizers.Adam(learning_rate=config.LR)
		mlm_model.compile(optimizer=optimizer)
		return mlm_model
		'''