'''
Sanjay Singh
san.singhsanjay@gmail.com
April-2021
To check training, validation and test data set
'''

# packages


# paths
train_data_path = "/home/sansingh/Downloads/Flickr8k_ImageCaptioning/archive/Flickr_8k.trainImages.txt"
test_data_path = "/home/sansingh/Downloads/Flickr8k_ImageCaptioning/archive/Flickr_8k.testImages.txt"
original_data_path = "/home/sansingh/Downloads/Flickr8k_ImageCaptioning/archive/captions.txt" 
validation_data_path = "/home/sansingh/Downloads/Flickr8k_ImageCaptioning/archive/Flickr_8k.valImages.txt"

# read training data
train_filenames = []
f_ptr = open(train_data_path, "r")
lines = f_ptr.readlines()
f_ptr.close()
for line in lines:
	train_filenames.append(line.strip())
print("Completed reading training filenames")

# read test data
test_filenames = []
f_ptr = open(test_data_path, "r")
lines = f_ptr.readlines()
f_ptr.close()
for line in lines:
	test_filenames.append(line.strip())
print("Completed reading test filenames")

# updating status
print("Number of images in training data: ", len(train_filenames))
print("Number of images in testing data: ", len(test_filenames))

# check if any image hasn't repeated in training and tresting data
duplicate_found = False
for i in range(len(test_filenames)):
	if(test_filenames[i] in train_filenames):
		print("Duplication found at index ", i, ", name: ", test_filenames[i])
		duplicate_found = True
if(duplicate_found == False):
	print("No duplicates found")

# read original data
original_filenames = set()
f_ptr = open(original_data_path, "r")
lines = f_ptr.readlines()
f_ptr.close()
for line in lines:
	line = line.split(",")[0]
	line = line.strip()
	original_filenames.add(line)
original_filenames = list(original_filenames)

# status of original data
print("Number of images in original data: ", len(original_filenames))

# check if all train and test images are available in available data or not
for i in range(len(train_filenames)):
	if(train_filenames[i] not in original_filenames):
		print("Train image not found, at index: ", i, ", name: ", train_filenames[i])
for i in range(len(test_filenames)):
	if(test_filenames[i] not in original_filenames):
		print("Test image not found, at index: ", i, ", name: ", test_filenames[i])
print("Completed checking if train and test filenames are in original dataset or not")

# extracting filenames for validation dataset
validation_filenames = []
f_ptr = open(validation_data_path, "w")
for i in range(len(original_filenames)):
	if(original_filenames[i] not in train_filenames and original_filenames[i] not in test_filenames):
		validation_filenames.append(original_filenames[i])
		f_ptr.write(original_filenames[i] + '\n')
f_ptr.close()
print("Completed writing validation filenames")

# status of validation filenames
print("Number of validation filenames: ", len(validation_filenames))
