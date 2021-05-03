# Flickr8k_ImageCaptioning  
## Introduction  
It is a Python project for generating captions for images.  
  
Following is an example of image captioning:  
![alt text](https://github.com/sansinghsanjay/Flickr8k_ImageCaptioning/blob/main/archive/Images/12830823_87d2654e31.jpg)  
  
Following are the possible captions for the above image:  
1. Children sit and watch the fish moving in the pond  
2. people stare at the orange fish  
3. Several people are standing near a fish pond  
4. Some children watching fish in a pool  
5. There are several people and children looking into water with a blue tiled floor and goldfish  
  
Following are some of the applications of Image Captioning:  
    
## Dataset  
Flickr8k dataset is used here.  
Following is the link of this dataset: [Flickr8k Image Captioning Dataset](https://www.kaggle.com/adityajn105/flickr8k)  
The above dataset has 8,091 images with a [captions.txt](https://github.com/sansinghsanjay/Flickr8k_ImageCaptioning/blob/main/archive/captions.txt) file mentioning five captions for each image.  
Thus, 8,091 images x 5 captions = 40,455 image-captions  or lines in [captions.txt](https://github.com/sansinghsanjay/Flickr8k_ImageCaptioning/blob/main/archive/captions.txt).  
Training file [Flickr_8k.trainImages.txt](https://github.com/sansinghsanjay/Flickr8k_ImageCaptioning/blob/main/archive/Flickr_8k.trainImages.txt) and Testing file [Flickr_8k.testImages.txt](https://github.com/sansinghsanjay/Flickr8k_ImageCaptioning/blob/main/archive/Flickr_8k.testImages.txt) are downloaded from the Internet (source missed). These training and testing files have name of images to be used in training and testing of model.  
  
Following are some sample images:    
![alt text](https://github.com/sansinghsanjay/Flickr8k_ImageCaptioning/blob/main/archive/images_for_readme/0.png) ![alt text](https://github.com/sansinghsanjay/Flickr8k_ImageCaptioning/blob/main/archive/images_for_readme/1.png)  
![alt text](https://github.com/sansinghsanjay/Flickr8k_ImageCaptioning/blob/main/archive/images_for_readme/2.png) ![alt text](https://github.com/sansinghsanjay/Flickr8k_ImageCaptioning/blob/main/archive/images_for_readme/3.png)  
![alt text](https://github.com/sansinghsanjay/Flickr8k_ImageCaptioning/blob/main/archive/images_for_readme/4.png) ![alt text](https://github.com/sansinghsanjay/Flickr8k_ImageCaptioning/blob/main/archive/images_for_readme/5.png)  
  
## Technology  
This project is at the intersection of following two technologies:  
1. Computer Vision (CV)  
2. Natual Language Generation (NLG)  
  
What is Natural Language Generation (NLG)?  
  
  
## Programming Language and Packages Used
The entire project is implemented in Python - 3.6.9.  
Following are the details of some of the crucial packages:  
1. Tensorflow:  
2. Keras:  
3. OpenCV2:  
  
## Scripts Execution Flow  
Script [check_training_val_test.py](https://github.com/sansinghsanjay/Flickr8k_ImageCaptioning/blob/main/scripts/check_training_val_test.py) verified that the name of images given in training and testing .txt file are in captions.txt or not. Alogn with this, this script has created a file [Flickr_8k.valImages.txt](https://github.com/sansinghsanjay/Flickr8k_ImageCaptioning/blob/main/archive/Flickr_8k.valImages.txt) which has names of images that are not in training and testing .txt file. These images can be used for validation, as the name of file suggests. At last, this script has generated and saved following plot which summarising the number of images in the entire dataset, training , validation and testing dataset.  
![alt text](https://github.com/sansinghsanjay/Flickr8k_ImageCaptioning/blob/main/archive/no_of_imgs_in_original_train_val_test.png)  
  
## Results Obtained  
  
## References  
  
