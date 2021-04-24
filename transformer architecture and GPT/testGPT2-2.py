# testGPT2-2.py
# author: Diego Magdaleno
# This program tests whether a machine can initialize the different
# GPT-2 architectures (does not test if each architecture can be
# trained.
# Tenorflow 1.14/1.15/2.4
# Python 3.7
# Windows/MacOS/Linux


import gc
import tensorflow as tf
import gpt2


#'''
# Configuration code for allowing GPU usage on Tensorflow 2. Comment
# out when running on Tensorflow 1 on CPU.
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
session = tf.compat.v1.Session(config=config)
#'''


def main():
	# Model hyperparameters.
	vocab_size = 20000 # Only consider the top 20K words.
	max_len = 80 # Maximum sequence size.
	embedding_size = 256 # Embedding size for each token.
	n_heads = 2 # Number of attention heads.
	ff_dim = 256 # Hidden layer size in feed forward neural network
	# inside transformer.

	# OpenAI GPT-2 Hyperparameters
	# Model name 		embed_dim	n_layers	n_heads
	# Small (124M)		768				12			12
	# Medium (355M)		1024			24			16
	# Large (774M)		1280			36			20
	# X-Large (1.5B)	1600			48			25
	# GPT-2 max_len or context_size = 1024
	# GPT-2 vocab_size = 50257
	# GPT-2 ff_dim = 4 * embeding_dim
	hparams = {
		"Small": {
			"vocab_size": 50257,
			"embedding_size": 786,
			"n_layers": 12,
			"n_heads": 12,
			"ff_dim": 786 * 4,
			"context_size": 1024
		},
		"Medium": {
			"vocab_size": 50257,
			"embedding_size": 1024,
			"n_layers": 24,
			"n_heads": 16,
			"ff_dim": 1024 * 4,
			"context_size": 1024
		},
		"Large": {
			"vocab_size": 50257,
			"embedding_size": 1280,
			"n_layers": 36,
			"n_heads": 20,
			"ff_dim": 1280 * 4,
			"context_size": 1024
		},
		"X-Large": {
			"vocab_size": 50257,
			"embedding_size": 1600,
			"n_layers": 48,
			"n_heads": 25,
			"ff_dim": 1600 * 4,
			"context_size": 1024
		}
	}
	loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

	# Iterate through each model and try to initialize each.
	model_status = {model: False for model in hparams}
	for model in hparams:
		new_gpt = gpt2.GPT2(n_heads=hparams[model]["n_heads"], 
							n_layers=hparams[model]["n_layers"], 
							vocab_size=hparams[model]["vocab_size"],
							ff_dim=hparams[model]["ff_dim"], 
							embedding_size=hparams[model]["embedding_size"], 
							context_size=hparams[model]["context_size"],
							loss=loss_fn)
		model_status[model] = True
		print("Model " + str(model) + " successfully initialized.")
		del new_gpt
		gc.collect()

	'''
	# Print out which models were able to be initialized.
	for model in model_status:
		print("Model " + str(model) + " successfully initialized: " +\
				str(model_status[model]))
	'''

	# Exit the program.
	exit(0)


if __name__ == '__main__':
	main()