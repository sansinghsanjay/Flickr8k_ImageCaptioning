# Flickr8k_ImageCaptioning  
## Introduction  
It is a Python project for generating captions for images.  
  
Automated Image Captioning can be defined as generating a textual description of a given image.  
  
This problem was well researched by Andrej Karpathy in his PhD at Stanford University.  
  
Deep Learning has achieved state-of-art result in Image Captioning. In this project, classic solution (naive approach, i.e. simple encoder-decoder based approach) for Image Captioning is implemented. An advanced solution for Image Captioning is Attention Mechanism which is also quite useful for Neural Machine Translation (i.e., translating text from one natural language to another natural language). To be more specific, Attention Mechanism for Image Captioning is called as Visual Attention Mechanism.  
  
Following is an example of image captioning:  
![alt text](https://github.com/sansinghsanjay/Flickr8k_ImageCaptioning/blob/main/archive/Images/12830823_87d2654e31.jpg)  
  
Following are the possible captions for the above image:  
1. Children sit and watch the fish moving in the pond  
2. people stare at the orange fish  
3. Several people are standing near a fish pond  
4. Some children watching fish in a pool  
5. There are several people and children looking into water with a blue tiled floor and goldfish  
  
Following are some of the applications of Image Captioning:  
1. Self Driving Cars  
2. Aid to the blind people: It can guide blind people by generating text for the scene in front and speaking it by using TTS (Text to Speech) systems.  
3. CCTV Cameras are everywhere, but along with viewing the world, it can generate relevant captions, then we can raise alarms as any malicious activity take place.  
4. Image Captioning can make Google Image Search better.
    
## Dataset  
Flickr8k dataset is used here.  
Following is the link of this dataset: [Flickr8k Image Captioning Dataset](https://www.kaggle.com/adityajn105/flickr8k)  
The above dataset has 8,091 images with a [captions.txt](https://github.com/sansinghsanjay/Flickr8k_ImageCaptioning/blob/main/archive/captions.txt) file mentioning five captions for each image. All captions are written by different different people for every image. The size of this dataset is 1.04 GB.  
Thus, 8,091 images x 5 captions = 40,455 image-captions  or lines in [captions.txt](https://github.com/sansinghsanjay/Flickr8k_ImageCaptioning/blob/main/archive/captions.txt).  
Training file [Flickr_8k.trainImages.txt](https://github.com/sansinghsanjay/Flickr8k_ImageCaptioning/blob/main/archive/Flickr_8k.trainImages.txt) and Testing file [Flickr_8k.testImages.txt](https://github.com/sansinghsanjay/Flickr8k_ImageCaptioning/blob/main/archive/Flickr_8k.testImages.txt) are downloaded from the Internet (source missed). These training and testing files have name of images to be used in training and testing of model.  
  
Following are some sample images:    
![alt text](https://github.com/sansinghsanjay/Flickr8k_ImageCaptioning/blob/main/archive/images_for_readme/0.png) ![alt text](https://github.com/sansinghsanjay/Flickr8k_ImageCaptioning/blob/main/archive/images_for_readme/1.png)  
![alt text](https://github.com/sansinghsanjay/Flickr8k_ImageCaptioning/blob/main/archive/images_for_readme/2.png) ![alt text](https://github.com/sansinghsanjay/Flickr8k_ImageCaptioning/blob/main/archive/images_for_readme/3.png)  
![alt text](https://github.com/sansinghsanjay/Flickr8k_ImageCaptioning/blob/main/archive/images_for_readme/4.png) ![alt text](https://github.com/sansinghsanjay/Flickr8k_ImageCaptioning/blob/main/archive/images_for_readme/5.png)  
  
## Technology  
In this project, classic solution (naive approach, i.e. simple encoder-decoder based approach) for Image Captioning is implemented. An advanced solution for Image Captioning is Attention Mechanism which is also quite useful for Neural Machine Translation (i.e., translating text from one natural language to another natural language). To be more specific, Attention Mechanism for Image Captioning is called as Visual Attention Mechanism.  
  
This project is at the intersection of following two technologies:  
1. Computer Vision (CV): To understand the content of a given image.  
2. Natual Language Generation (NLG): NLG transforms data into plain English text. It is a branch of Natural Language Processing (NLP). 
  
## Applications of Natural Language Generation (NLG)  
Following are some of the applications of NLG:  
1. Freeform Text Generation: User provides an input, like a phrase, sentence or paragraph and the NLG model generates continuation of this input as output. For instances, Google Smart Compose predicts a phrase following a word input in Gmail.  
2. Question Answering: This is a system that can answer questions posed by humans. These systems can be open ended or closed ended (domain specific).  
3. Summarization: Summarization reduces the amount of information while capturing the most important details in a narrative. This is of two types:  
  i. Extractive Summarization: It takes the most important phrases or sentences and stitches them together to form a summarizated narrative.  
  ii. Abstractive Summarization: This is equivalent of a human writing a summary in his / her own words. For instance, headline generation, abstract for journals / whitepaper / etc.  
4. Image Captioning  
  
### How NLG is different from NLP  
NLP is focussed on deriving analytic insights from textual data. Whereas, NLG is used to synthesize textual content by combining analytic output with contextualized narratives. In short, NLP reads while NLG writes.  
  
## Programming Language and Packages Used
The entire project is implemented in Python - 3.6.9.  
Following are the details of some of the crucial packages:  
1. Tensorflow:  
2. Keras:  
3. OpenCV2:  
  
## Scripts Execution Flow  
Script [check_training_val_test.py](https://github.com/sansinghsanjay/Flickr8k_ImageCaptioning/blob/main/scripts/check_training_val_test.py) verified that the name of images given in training and testing .txt file are in captions.txt or not. Alogn with this, this script has created a file [Flickr_8k.valImages.txt](https://github.com/sansinghsanjay/Flickr8k_ImageCaptioning/blob/main/archive/Flickr_8k.valImages.txt) which has names of images that are not in training and testing .txt file. These images can be used for validation, as the name of file suggests. At last, this script has generated and saved following plot which summarising the number of images in the entire dataset, training , validation and testing dataset.  
![alt text](https://github.com/sansinghsanjay/Flickr8k_ImageCaptioning/blob/main/archive/no_of_imgs_in_original_train_val_test.png)  
  
## Evaluation Metric Used Here: BLEU Score  
Evaluating NLG systems is a much more complicated task. There are following four evaluation metrics for evaluating a NLG system:  
1. Bilingual Evaluation Understudy (BLEU Score)  
2. Recall Oriented Understudy for Gisting Evaluation (ROUGE)  
3. Metric for Evaluation for Translation with Explicit Ordering (METEOR)  
4. Consensus based image Descriptive Evaluation (CIDEr)  
Since above metrics differ mostly in terms of precision and recall (i.e., sensitivity), thus we will first see how to calculate precision and recall (i.e., sensitivity) in NLG.  
In general,  
&nbsp;&nbsp;&nbsp;&nbsp;Precision is the ratio of number of correctly predicted positive instances out of total number of predicted positive instances.  
  ![alt txt](https://github.com/sansinghsanjay/Flickr8k_ImageCaptioning/blob/main/maths_eqn/precision_eq.gif)  
&nbsp;&nbsp;&nbsp;&nbsp;Recall or Sensitivity is the number of correctly predicted positive instances out of total number of actual positive instances.  
  Eq of Recall (or Sensitivity)  
In NLG:  
&nbsp;&nbsp;&nbsp;&nbsp;We call predicted (or generated) text as candidate text and actual text as reference text. Consider the following case:  
&nbsp;&nbsp;&nbsp;&nbsp;Reference: "I work on machine learning"  
&nbsp;&nbsp;&nbsp;&nbsp;Candidate A: "I work"  
&nbsp;&nbsp;&nbsp;&nbsp;Candidate B: "He works on machine learning"  
&nbsp;&nbsp;&nbsp;&nbsp;Precision in NLG: Eq  
&nbsp;&nbsp;&nbsp;&nbsp;Recall (or sensitivity) in NLG: Eq  
&nbsp;&nbsp;&nbsp;&nbsp;Precision of Candidate A: Eq  
&nbsp;&nbsp;&nbsp;&nbsp;Recall of Candidate A: Eq  
&nbsp;&nbsp;&nbsp;&nbsp;Precision of Candidate B: Eq  
&nbsp;&nbsp;&nbsp;&nbsp;Recall of Candidate B: Eq  
&nbsp;&nbsp;&nbsp;&nbsp;Above calculations are done by using unigram. One can also use bigram, trigram and so on (i.e., n-gram) and result will different.  
&nbsp;&nbsp;&nbsp;&nbsp;In modified n-gram precision scheme, we match candidate's n-gram only as many times as they are present in any of reference's text.  
&nbsp;&nbsp;&nbsp;&nbsp;Finally, to include all the n-gram precision scores in our final precision, we take their geometric mean. This is because it has been found that precision decreases exponentially with n; and as such we would require logarithmic averaging to represent all values fairly.  
&nbsp;&nbsp;&nbsp;&nbsp;Precision Eq  
  
## Results Obtained  
  
## References  
  
