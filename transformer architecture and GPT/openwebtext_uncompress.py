# openwebtext_uncompress.py
# author: Diego Magdaleno
# This program is meant to uncompress the contents of the OpenWebText
# Corpus. There are two possible soures to get the content. This
# program deals with the first source. After downloading the tar.xz
# file from their link, this program will uncompress that file to
# extract all files to another folder. It will then uncompress all
# files in the folder to txt files to be used for training GPT. Note
# that this is not the exact same openwebtext corpus and GPT-2 that
# OpenAI has created.
# Source: https://skylion007.github.io/OpenWebTextCorpus/
# Source(alt): https://github.com/jcpeterson/openwebtext
# Python 3.7
# Windows/MacOS/Linux


import os
import lzma
import tarfile
from tqdm import tqdm


def main():
	# Locate the openwebtext.tar.xz file. File must be in the same
	# current working directory as the program.
	print("Searching for openwebtext.tar.xz...")
	if not os.path.exists("openwebtext.tar.xz"):
		print("Error: Required file openwebtext.tar.xz missing.")
		exit(1)
	print("openwebtext.tar.xz found.")

	# Create folder to store the resulting files from the tar.xz file.
	if not os.path.exists("openwebtext"):
		os.mkdir("openwebtext")

	# Uncompress the tar.xz file.
	print("Uncompressing openwebtext.tar.xz...")
	tar = tarfile.open("openwebtext.tar.xz", "r:xz")
	tar.extractall()
	tar.close()
	print("openwebtext.tar.xz uncompressed.")

	# Go through each file and uncompress it from .xz to .txt.
	compressed_files = [file for file in os.listdir("./openwebtext")
						if file.endswith(".xz")]
	print("Uncompressing each file...")
	for file in tqdm(range(len(compressed_files))):
		# Read compressed file contents with lzma module.
		open_file = lzma.open("./openwebtext/" + compressed_files[file], "rb")
		#file_lines = open_file.readlines()
		file_contents = open_file.read()
		open_file.close()

		# Write file contents from compressed file into a .txt file
		# with the same name.
		new_file = open("./openwebtext/" + compressed_files[file].rstrip(".xz") + ".txt", "wb+")
		new_file.write(file_contents)
		new_file.close()

		# Clean file contents (remove special string found in texts).
		clean_file_text("./openwebtext/" + compressed_files[file].rstrip(".xz") + ".txt")
	'''
	# Same code except it does not require the use of the tqdm module.
	for file in compressed_files:
		open_file = lzma.open("./openwebtext/" + file, "rb")
		#file_lines = open_file.readlines()
		file_contents = open_file.read()
		open_file.close()

		new_file = open("./openwebtext/" + file.rstrip(".xz") + ".txt", "wb+")
		new_file.write(file_contents)
		# new_file.close()

		# # Clean file contents (remove special string found in texts).
		clean_file_text("./openwebtext/" + compressed_files[file].rstrip(".xz") + ".txt")
	'''
	print("Files uncompressed.")

	# Clean (remove) the folder.
	print("Cleaning directory...")
	for file in compressed_files:
		os.remove("./openwebtext/" + file)
	print("Directory clean.")

	# Exit the program.
	print("OpenWebText Corpus successfully extracted.")
	exit(0)


# Clean the text file specified by the argument. 
# @param: file_path, a string that specifies the file path to the text
#	that needs to be cleaned up.
# @return: returns nothing.
def clean_file_text(file_path):
	# Open file specified by the argument and read its contents to a
	# list.
	with open(file_path, "r", encoding="utf-8") as read_file:
		file_lines = read_file.readlines()

	# There are special strings in the text. These may interfere
	# with tokenization/encoding and need to be removed.
	# Examples:
	# 0107725-5c1cfcbb66068a2e76d1b7d3350adc2a.txt0000644000000000000000000002364700000000000015324 0ustar  00000000000000
	# 0107919-6e84ed348c613c3f24ac6999d4fdbcfc.txt0000644000000000000000000000705000000000000015362 0ustar  00000000000000
	# String length is 116. Magic string value in all is the
	# "ustar". Check for these special strings in a line. Use the
	# "\x00" to guage where to splice the string. This will remove
	# both the "\x00" and the long unique string from the texts.
	for line in range(len(file_lines)):
		# Use list splicing to remove the special string if it's
		# found in that line.
		if "\x00" in file_lines[line] and "ustar" in file_lines[line]:
			start_index = file_lines[line].index("\x00")
			end_index = file_lines[line][::-1].index("\x00")
			file_lines[line] = file_lines[line][:start_index] + " " + file_lines[line][-end_index:]
		# If the line does not have the special string, but still
		# has the "\x00" value, just remove all instances of that
		# value from the line.
		elif "\x00" in file_lines[line]:
			file_lines[line] = file_lines[line].replace("\x00", "")

	# Remove empty lines that are just "\n"
	while "\n" in file_lines:
		file_lines.remove("\n")

	# Write the changes to the file.
	with open(file_path, "w", encoding="utf-8") as write_file:
		write_file.write("\n".join(file_lines))

	# Return the function.
	return


if __name__ == '__main__':
	main()