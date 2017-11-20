**Behavioral Cloning** 
---
## Overview
In this project for Udacity's Self Driving Car Nano Degree Program, we explore end-to-end learning for self driving cars.  In a simulated environment, we teach how a car to drive using deep learning. Specifically, a deep neural network is implemented to learn the mapping from the camera data to steering angles.

**Discussion**
--

## Exploring Driving Data
First, I explored how the data is distributed for steering angle. As one can see from the graph above, it can be seen that the steering command values around 0 degrees have a dominating presence in the dataset:

<img src="writeup_images/original_data_distribution.png" width="480" alt="Input Image 1" />

In machine learning, it is generally known fact that scewed data set makes it hard to build robust models. Within the context of self-driving cars, I theoretized that if the neural network is learned to drive straight most of the time, it will probably drive straight if it encounters unexpected scenes while driving. This was behaviour was verified while testing. Whenever the car failed to make a sharp turn, it was most of the time because it kept insisting on driving straight. I subsampled the dataset by randomly dropping any data points with steering angle value less than 0.05 degrees and their corresponding input images. The result is more balanced dataset:

<img src="writeup_images/subsampled_data.png" width="480" alt="Input Image 1" />

## Driving Data Augmentation
While exploring the dataset, it became clear that the dataset has to be augmented. There are few images with shawdows on the road but if the car were to learn what to do when it sees sahdows, it would need more to handle such situations. I augmented the data with random brightness changes to help with this issue:

<img src="writeup_images/change_brightness.png" width="480" alt="Input Image 1" />

It was also mentioned in the course how the driving course is designed to turn left most of the time and the car will not be able to learn good driving habbits to accomodate for the right turns. To mitigate this issue some of the images and settering angles were selected to be flipped horizontally:

<img src="writeup_images/flip.png" width="480" alt="Input Image 1" />

Also, it is expected that the neural network will not be able to perfectly copy the expert's behaviour from the training dataset. This will mean that the car will run into situations that is not recored in the origianl training dataset. In order to build a robust driving model, I added random translation to the input images and modified the steering angles accordingly in order to simulate the car driving on many different positions and orientations on the road. Some of the examples of this augmentations are shown below.

<img src="writeup_images/translation1.png" width="480" alt="Input Image 1" />

<img src="writeup_images/translation2.png" width="480" alt="Input Image 1" />

## More Data
After augmentation, I was able to get the car to drive the full course but on sharp turns, the deep neural network would do something that would not be considered safe if humans were the drivers. I added one lap of data for those sharp turns to ensure that the network learns proper policies to make those turns. In the end, I ended up with 9000 data points for training a network that successfully drives around the course.

## Training/Validation/Test Split
I splitted the given data into training, validation and test data in 80/10/10 ratio to make sure I can verify my experiments with the deep neural net arcitecture and data augmentation/composition.

## Deep Neural Net Arcitecture 
For my project, I decided to use the NVIDIA model as shown in the digram below:

<img src="writeup_images/nVidia_model.png" width="480" alt="Input Image 1" />

Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
lambda_1 (Lambda)                (None, 160, 320, 3)   0           lambda_input_1[0][0]             
____________________________________________________________________________________________________
cropping2d_1 (Cropping2D)        (None, 65, 320, 3)    0           lambda_1[0][0]                   
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 31, 158, 24)   1824        cropping2d_1[0][0]               
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 14, 77, 36)    21636       convolution2d_1[0][0]            
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 5, 37, 48)     43248       convolution2d_2[0][0]            
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 3, 35, 64)     27712       convolution2d_3[0][0]            
____________________________________________________________________________________________________
convolution2d_5 (Convolution2D)  (None, 1, 33, 64)     36928       convolution2d_4[0][0]            
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 2112)          0           convolution2d_5[0][0]            
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 100)           211300      flatten_1[0][0]                  
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 50)            5050        dense_1[0][0]                    
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 10)            510         dense_2[0][0]                    
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 1)             11          dense_3[0][0]                    
====================================================================================================
Total params: 348,219
Trainable params: 348,219
Non-trainable params: 0
____________________________________________________________________________________________________


## Training
I used Adam optimization as the gradient descent alogirthm and Mean Squared Error as the cost function for training. After few trials, I found out that Adam optimizer is better to be trained with a learning rate of 1e-4, instead of the default 1e-3. I also discovered that 5 epochs is enough to train a robust model.
