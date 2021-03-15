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

docker build -t buildTagName Path/To/Docker/File

To run the docker image, run the following command:

docker run -it --rm --volume volume/path/on/container buildTagName
