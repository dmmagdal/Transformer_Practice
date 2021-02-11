# convoai.py
# A simple conversational AI that can be trained or used out-of-the-box
# using the huggingface transformers API.
# source: https://towardsdatascience.com/how-to-train-your-chatbot-
# with-simple-transformers-da25160859f4
# https://datamahadev.com/building-a-chatbot-in-python/
# Python 3.7
# Winows/MacOS/Linux


import os
from simpletransformers.conv_ai import ConvAIModel


train_args = {
	"overwrite_output_dir": True,
	"reprocess_input_data": True
}

# Create a ConvAIModel and load the transformer with pretrained
# weights. The ConvAIModel comes with a lot of configuration options
# which can be found in its documentation.
model = ConvAIModel("gpt", "gpt_personachat_cache", use_cuda=True, args=train_args)

# Train the model on the Persona-Chat training data. To train the model
# on your own custom data, you must create a JSON file with the
# following structure (see train_example.json). This structure is used
# in the Persona-Chat dataset.
model.train_model()

# Each entry in Persona-Chat is a dict with two keys: personality and
# utterances, and the dataset is a list of entries. For personality, it
# is a list of strings containing the personality of the agent. For
# utterances, it is the list of dictionaries, each of which has two
# keys which are lists of strings. The candidates are a list of strings
# structured as such: [next_utterance_candidate_1, ..., 
# next_utterance_candidate_19]. The last candidate is the ground truth
# response observedin the conversational data. The history is also a
# list of strings structured as such: [dialog_turn_0, ..., 
# dialog_turn_n] where n is an odd number since the other user starts
# every conversation.
# For preprocessing, spaces before periods at end of sentences and
# everything lowercase.
# Assuming the custom training data is in a json file with the given
# structure and saved in data/train.json, you can train the model by
# executing model.train_model("data/minimal_train.json").

# Evaluation can be performed on the Persona-Chat dataset just as
# easily by calling the eval_model() method. As with training, you may
# provide a different evaluation dataset as long as it follows the 
# correct structure.
model.eval_model()

# To talk with the model, simply call model.interact(). This will pick
# random personality from the dataset and let you talk with it from
# the terminal.
model.interact()

# Alternatively, you can create a personality on the fly by giving the
# interact() method a list of strings to build a personality from.
model.interact(
	personality=[
		"i like computers .",
		"i like reading books .",
		"i love classical music .",
		"i am very outgoing ."
	]
)

# To load a trained model, you need to provide the path to the
# directory containing the model file when creating the ConvAIModel
# object.
model = ConvAIModel("gpt", "outputs")