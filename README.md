# Simpsonify
Simpsonify is a simple project in Tensorflow dedicated to train Generative Adversarial Network to transfer Simpsons 
cartoon style into your profile picture!  

I was inspired by excellent [CycleGan tutorial](https://www.tensorflow.org/tutorials/generative/cyclegan)
written by people from Tensorflow. I have used two datasets - [Simpsons Faces](https://www.kaggle.com/kostastokis/simpsons-faces)
and [LFW - People (Face Recognition)](https://www.tensorflow.org/tutorials/generative/cyclegan). For training 9000 samples
from each dataset are taken. I'm testing an algorithm using my own photo, so it is also included in the repository in
scripts/data/zdjecie.jpg

##How to use it?  
Actually whole pipeline can be used for different CycleGAN datasets - fameous painters style transfer etc. 
There are simple steps to start training:
1. Run scripts/data/generate_metafile.py with *--data_path* parameter which is the path where the dataset with two
classes subfolders is stored. As a result you get *metafile.csv* file with paths to every picture in the dataset. You
only have to do this once.
2. Run scripts/train/train_cycle_gan.py with two parameters: *--metafile_path* which is path to the metafile generated
in the first step, *checkpoint_path* which is path where training checkpoints during the training are saved. As I am
using Google Colab for training it is very useful due to frequent disconnections.

I have also included notebook which I used in Google Colab for training - scripts/train/simpsons.ipynb

If you have any questions feel free to contact me! kowaleczko.p@gmail.com
