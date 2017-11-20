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

Also, it is expected that the neural network will not be able to perfectly copy the expert's behaviour from the training dataset. This will mean that the car will run into situations that is not recored in the origianl training dataset. In order to build a robust driving model, I added random translation to the input images and modified the steering angles accordingly in order to simulate the car driving on many different positions and orientations on the road.
<img src="writeup_images/translation1.png" width="480" alt="Input Image 1" />

<img src="writeup_images/translation2.png" width="480" alt="Input Image 1" />

