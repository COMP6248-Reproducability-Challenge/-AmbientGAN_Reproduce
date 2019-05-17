# AmbientGAN_Reproduce
This repository provides code to reproduce results from the paper AmbientGAN: Generative models from lossy measurements.

    This repository is a work under COMP6248 Reproducability Challenge project. 
    The code is mainly based on https://github.com/shinseung428/ambientGAN_TF/blob/master/ambientGAN.py. This is a Tensorflow 
    implementation of AmbientGAN but for celeA dataset only.
    Our contribution: 
    1. Support Mnist dataset and related use cases. While following the framework of the above code, we made big changies in model 
    and corresponding functions.
    2. For experiments on Celeb dataset, we kept the framework of the above code but made lots of modifications in order to fit our 
    designs. To be specific, we changed the structure of the generator and discriminator in ambientGAN.py file in order to have better 
    performance. Moreover, we changed some hard coded configurations to fit our experiment and added some new functions to output the 
    result images we want.
    Also we refered the implementation of original paper https://github.com/AshishBora/ambient-gan for a few tool functions.
    

    Requirements:
    Python 2.7
    Tensorflow
    matplotlib
    scipy
    numpy
    cvxpy
    scikit-learn
    tqdm
    opencv-python
    pandas
    
    Dataset:
    Mnist
    CelebA
    
    Run:
    for Mnist:
    Load the whole folder
    change the data path in ops.py
    run train
    
    for celeb color:
    This folder includes codes for section 2.2.2 in the reproducing paper
    Follow the steps below to run code in this folder.
    For generating results of Ambient gan
    1. Put the dataset img_align_celeba in the folder ./data
    2. Run “train.py”
    For generating results of baseline (Pure DCGAN training with measured and recovered samples)
    1. Put the dataset img_align_celeba in the folder ./data
    2. Replace the sentence in “ambientGAN.py”
       “self.Y_g = self.measurement_fn(self.X_g, name="measurement_fn”)”
       With
       “self.Y_g = self.X_g”
    3. Run “train.py”
    
    for celeb grey:
    This folder includes codes for section 2.2.1 in the reproducing paper
    Follow the steps below to run code in this folder.
    For generating results of Ambient gan
    1. Put the dataset img_align_celeba in the folder ./data
    2. Run “train.py”
    For generating results of baseline (Pure DCGAN)
    1. Put the dataset img_align_celeba in the folder ./data
    2. Replace the sentence in “ambientGAN.py”
       “self.Y_g = self.measurement_fn(self.X_g, name="measurement_fn”)”
       With
       “self.Y_g = self.X_g”
    3. Run “train.py”
    
    Mnist result
![image](https://github.com/RickRe/AmbientGAN_Reproduce/blob/master/images/p1.png)
    block pixel probability=0.5
![image](https://github.com/RickRe/AmbientGAN_Reproduce/blob/master/images/p2.png)
    hybrid measurement
    
    Celeb result
![image](https://github.com/RickRe/AmbientGAN_Reproduce/blob/master/images/crop1.jpg)

    crop1
    
![image](https://github.com/RickRe/AmbientGAN_Reproduce/blob/master/images/crop2.jpg)

    crop2
    
![image](https://github.com/RickRe/AmbientGAN_Reproduce/blob/master/images/crop3.jpg)

    crop3
    
![image](https://github.com/RickRe/AmbientGAN_Reproduce/blob/master/images/crop4.jpg)

    crop4
    
![image](https://github.com/RickRe/AmbientGAN_Reproduce/blob/master/images/crop5.jpg)

    crop5
    
![image](https://github.com/RickRe/AmbientGAN_Reproduce/blob/master/images/crop6.jpg)

    crop6
    
![image](https://github.com/RickRe/AmbientGAN_Reproduce/blob/master/images/crop7.jpg)

    crop7
    
![image](https://github.com/RickRe/AmbientGAN_Reproduce/blob/master/images/crop8.jpg)

    crop8
    
![image](https://github.com/RickRe/AmbientGAN_Reproduce/blob/master/images/crop9.jpg)

    crop9
    
![image](https://github.com/RickRe/AmbientGAN_Reproduce/blob/master/images/crop10.jpg)

    crop10
    
![image](https://github.com/RickRe/AmbientGAN_Reproduce/blob/master/images/crop14.jpg)

    crop14
    
![image](https://github.com/RickRe/AmbientGAN_Reproduce/blob/master/images/crop15.jpg)

    crop15
    
![image](https://github.com/RickRe/AmbientGAN_Reproduce/blob/master/images/crop16.jpg)

    crop16
    
![image](https://github.com/RickRe/AmbientGAN_Reproduce/blob/master/images/crop17.jpg)

    crop17


