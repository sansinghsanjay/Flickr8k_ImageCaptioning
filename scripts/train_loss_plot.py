'''
Sanjay Singh
san.singhsanjay@gmail.com
May-2021
To create plot of loss values generated while training
'''

# packages
import matplotlib.pyplot as plt

# paths
train_log_file_path = "/home/sansingh/github_repo/Flickr8k_ImageCaptioning/output/trained_models/train_logs.txt"
target_path = "/home/sansingh/github_repo/Flickr8k_ImageCaptioning/output/trained_models/"

# reading train log
f_ptr = open(train_log_file_path, "r")
lines = f_ptr.readlines()
f_ptr.close()
loss_values = list() 
for line in lines: 
	line = line.strip() 
	tokens = line.split(' ') 
	loss_values.append(float(tokens[len(tokens) - 1]))

# making plot
plt.title("Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss Value")
plt.grid()
plt.plot(loss_values)
plt.savefig(target_path + "train_loss_plot.png")

