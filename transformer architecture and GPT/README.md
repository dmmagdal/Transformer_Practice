# The GPT2 implementation witk tensorflow/keras.

### About the module

The main module gpt2.py is capable of initializing the GPT2 model from OpenAI and is based on
the tutorial from the Keras website. The size of the model created is limited by the
computing resources of the host machine. The files testGPT2.py and testGPT2-2.py are simple
programs that test out the capabilities of the gpt2.py module.

###

### Docker

To run the program on Docker, there are two docker files for CPU only and GPU. The docker file
uses Tensorflow 2.4.0 and can be built and run with the following commands:

docker build -t buildTagName -f Path/To/Docker/File .

e.g.

docker build -t gpt-2-gpu -f Dockerfile.gpu .

OR

docker build -t gpt-2-cpu -f Dockerfile.cpu .

To run the docker image, run the following command:

docker run -it --rm --volume volume/path/on/container buildTagName

### Docker resources

Follow the following links for running Tensorflow in Docker:

	Tensorflow website:
https://www.tensorflow.org/tfx/serving/docker
https://www.tensorflow.org/install/docker

	Dockerhub website:
https://hub.docker.com/r/tensorflow/tensorflow/
https://hub.docker.com/r/tensorflow/tensorflow/tags/?page=1&ordering=last_updated&name=2.4.0

	Docker cheatsheet:
https://groupe-sii.github.io/cheat-sheets/docker/index.html

### Tensorflow Text Encoders

TextVectorization Layer
https://www.tensorflow.org/api_docs/python/tf/keras/layers/experimental/preprocessing/TextVectorization

SubwordTextEncoder
https://www.tensorflow.org/datasets/api_docs/python/tfds/deprecated/text/SubwordTextEncoder

ByteTextEncoder
https://www.tensorflow.org/datasets/api_docs/python/tfds/deprecated/text/ByteTextEncoder

Tokenizer
https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/text/Tokenizer