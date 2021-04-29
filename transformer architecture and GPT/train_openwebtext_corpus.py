# train_openwebtext_corpus.py
# author: Diego Magdaleno
# Train a GPT2 model (from gpt2.py) on the openwebtextcorpus and save it.
# Should be very similar to testGPT2.py.
# Tensorflow 2.4
# Python 3.7
# Windows/MacOS/Linux


import os
import string
import random
import gpt2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization


'''
# Configuration code for allowing GPU usage on Tensorflow 2. Comment
# out when running on Tensorflow 1 on CPU.
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
session = tf.compat.v1.Session(config=config)
'''


def main():
	# Index all the files from the openwebtext corpus.
	print("Indexing training files...")
	corpus_path = "./openwebtext"
	text_files = [corpus_path + "/" + file 
					for file in os.listdir(corpus_path)]
	batch_size = 128
	print("All training data indexed.")

	# Create a dataset from the text fields.
	print("Initializing dataset...")
	random.shuffle(text_files)
	text_ds = tf.data.TextLineDataset(text_files)
	text_ds = text_ds.shuffle(buffer_size=256)
	text_ds = text_ds.batch(batch_size)
	print("Dataset initialized.")

	# Model hyperparameters.
	vocab_size = 50257
	context_size = 1024
	embedding_size = 1024
	n_heads = 16
	n_layers = 16
	ff_dim = 1024 * 4

	# Note that these model parameters will create a model from gpt2.py
	# with roughly 1.3 Billion parameters. The default model created
	# from gpt2.py (no hyperparameters set, uses default arguments)
	# creates a model with roughly 740 Million parameters.
	# Ammendment to Note: While the Dell Desktop has no problem
	# initializing this version of the dataset, it runs out of memory
	# attempting to train it. Will have to use smaller model
	# configurations (ie, less layers & heads).
	# Next ammendment: Unable to train model with default
	# hyperparameters. Still run out of memory when trying to train
	# model.
	n_heads = 9
	n_layers = 9

	# These hyperparemeters give a model with roughly 520 Million 
	# parameters. Still runs out of memory on training.
	n_layers = 6
	n_heads = 6
	
	# These hyperparemeters give a model with roughly 305 Million 
	# parameters. Still runs out of memory on training.
	n_layers = 1
	n_heads = 4
	embedding_size = 768
	ff_dim = 768 * 4

	n_layers = 1
	n_heads = 2
	ff_dim = 256
	vocab_size = 20000
	embedding_size = 256
	context_size = 80

	# Mapping hyperparameters to models and their abilitiy to compile.
	# model_name, parameters, n_layers, n_heads, ff_dim, vocab_size, embedding_size, context_size, compile_model, run_model, reason_to_not_run
	# gpt2_xxs    11 Million  1         2        256     20,000      256             80            Yes            Yes        N/A
	# gpt2_xs     92 Million  1         4        3072    50,257      768             1024          Yes            No         Ran out of memory on first Decoder (MultiHeadAttention) layer
	# gpt2_small  305 Million 6         6        4096    50,257      1024            1024          Yes            No         Ran out of memory on first Decoder (MultiHeadAttention) layer
	# gpt2_medium 502 Million 9         9        4096    50,257      1024            1024          Yes            No         Ran out of memory on first Decoder (MultiHeadAttention) layer
	# gpt2_large  740 Million 12        12       32      65,536      1024            1024          Yes            No         Ran out of memory on TokenizationAndPosition layer
	# gpt2_xl     1.3 Billion 16        16       4096    50,257      1024            1024          Yes            No         Ran out of memory on TokenizationAndPosition layer
	# Again, gpt2_large is the same as using the default
	# hyperparameters/arguments on the GPT2 class in gpt2.py.

	# Set all text to lowercase and handle punctuation.
	# @param: input_string, the input string of the data.
	# @return: returns a cleaned version of the string.
	def custom_standardization(input_string):
		lowercased = tf.strings.lower(input_string)
		stripped_html = tf.strings.regex_replace(lowercased, "<br />", " ")
		return tf.strings.regex_replace(stripped_html, f"([{string.punctuation}])", r" \1")

	def prepare_lm_inputs_labels(text):
		text = tf.expand_dims(text, -1)
		tokenized_sentences = vectorize_layer(text)
		x = tokenized_sentences[:, :-1]
		y = tokenized_sentences[:, 1:]
		return x, y

	# Create a vectorization layer and adapt it to the text.
	print("Creating text vectorization and cleaning dataset...")
	vectorize_layer = TextVectorization(
		standardize=custom_standardization,
		max_tokens=vocab_size - 1,
		output_mode="int",
		output_sequence_length=context_size + 1
	)
	vectorize_layer.adapt(text_ds)
	vocab = vectorize_layer.get_vocabulary() # Get words back from token indices
	text_ds = text_ds.map(prepare_lm_inputs_labels)
	text_ds = text_ds.prefetch(tf.data.experimental.AUTOTUNE)
	print("Done.")

	# Training hyperparameters.
	# Verbosity. 0 => silent, 1 => progress bar, 2 => 1 line per epoch.
	#verbose = 2
	verbose = 1
	epochs = 25

	# Tokenize starting prompt.
	print("Tokenizing starting prompt...")
	word_to_index = {}
	for index, word in enumerate(vocab):
		word_to_index[word] = index
	start_prompt = "good morning my friends"
	start_tokens = [word_to_index.get(_, 1) for _ in start_prompt.split()]
	num_tokens_generated = 256
	text_gen_callback = gpt2.TextGenerator(num_tokens_generated, start_tokens, 
											vocab, context_size)
	print("Created text generator callback.")

	# Initialize a GPT2 object.
	print("Initializing GPT2 model...")
	loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
	new_gpt = gpt2.GPT2(n_heads=n_heads, n_layers=n_layers, vocab_size=vocab_size,
						ff_dim=ff_dim, embedding_size=embedding_size, 
						context_size=context_size, loss=[loss_fn, None], 
						metrics=None, model_name="gpt2-xxs")
	'''
	new_gpt = gpt2.GPT2(n_heads=n_heads, n_layers=n_layers, vocab_size=vocab_size,
						ff_dim=ff_dim, embedding_size=embedding_size, 
						context_size=context_size, loss=[loss_fn, None], 
						metrics=None, model_name="gpt2-xs")
	'''
	'''
	new_gpt = gpt2.GPT2(n_heads=n_heads, n_layers=n_layers, vocab_size=vocab_size,
						ff_dim=ff_dim, embedding_size=embedding_size, 
						context_size=context_size, loss=[loss_fn, None], 
						metrics=None, model_name="gpt2-small")
	'''	
	'''
	new_gpt = gpt2.GPT2(n_heads=n_heads, n_layers=n_layers, vocab_size=vocab_size,
						ff_dim=ff_dim, embedding_size=embedding_size, 
						context_size=context_size, loss=[loss_fn, None], 
						metrics=None, model_name="gpt2-medium")			
	'''
	'''
	new_gpt = gpt2.GPT2(vocab_size=vocab_size, 
						loss=[loss_fn, None],
						model_name="gpt2-large")
	'''
	'''
	new_gpt = gpt2.GPT2(n_heads=n_heads, n_layers=n_layers, vocab_size=vocab_size,
						ff_dim=ff_dim, embedding_size=embedding_size, 
						context_size=context_size, loss=[loss_fn, None], 
						metrics=None, model_name="gpt2-xl")
	'''
	print("Done.")

	# Train GPT2 on data.
	print("Starting training...")
	if not os.path.exists("./model_checkpoints"):
		os.mkdir("./model_checkpoints")
	model_checkpoint = keras.callbacks.ModelCheckpoint(
		"./model_checkpoints/", save_weights_only=True, save_best_only=True
	)
	new_gpt.gpt_model.fit(
		text_ds, verbose=verbose, epochs=epochs, batch_size=64
		callbacks=[text_gen_callback, model_checkpoint]
	)
	print("Model trained.")

	# Save GPT2 model.
	#save_path = "./gpt2_xl"
	#save_path = "./gpt2_large"
	#save_path = "./gpt2_medium"
	#save_path = "./gpt2_small"
	#save_path = "./gpt2_xs"
	save_path = "./gpt2_xxs"
	if not os.path.exists(save_path):
		os.mkdir(save_path)
	new_gpt.save(save_path)

	# Load GPT2 model.
	load_gpt = gpt2.GPT2()
	load_gpt.load(save_path)
	print(load_gpt.gpt_model.summary())

	# Exit the program.
	exit(0)


if __name__ == '__main__':
	main()