## Advanced Lane Finding

This project is about building a lane detector using more advanced computer vision algorithms.

Overview
---

The goals of the project are the following:
* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply the distortion correction to the raw image.  
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view"). 
* Detect lane pixels and fit to find lane boundary.
* Determine curvature of the lane and vehicle position with respect to center.
* Warping the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.
* Run the entire pipeline on a sample video recorded on a sunny day on the I-280. 

Most of the algorithm architecture is provided by the udacity course. The key for the successful implementation depended a lot on parameter tuning.

## Pipeline

Camera Calibration
---

Camera calibration step involves calibrating out the distortion caused by the camera lens using a set of chessboard images. Calibration images are provided under camera_cal directory. 

I started by preparing a list of "object points" that represent expected coordinates (x, y, z) of corners in the chessboard patterns in the world coordinate space, with z assumed to be 0 for all the points. The goal here is to provide the calibration algorithm what the chessboard corners should look like in a 2d plane in reality. 

Next, I used OpenCV's `findChessboardCorners()` function which can detect a chessboard pattern given an image and number of corners in horizontal and vertical directions. The chessboard patterns contain 9 and 6 corners in the horizontal and vertical directions. `findChessboardCorners()` function outputs "image points" coordinates of the corners in the camera's view.

The object points and image points then can be fed into the `calibrateCamera()` function which computes the calibration parameters for correcting the camera lens distortion.

Here is an example that demonstrates the camera calibration procedure using a chessboard pattern image:
<img src="./output_images/camera_calibration.png" />

Perspective Transform
---

Because cameras project 3d objects into 2d image plane as it captures images, it becomes hard to uncover the actual shape of the lanes in 3d space. To solve this problem, perspective transform can be used to transform the original incoming image so that it looks like the image was taken from the top view.

In order to do the transform, we need to specify source points in the original image coordinate and corresponding destination points in the transformed coordinate. I had to perform a series of trial and error to find out points for the bounding box that would be used for the perspective transform. Figuring out the coordinates of the bounding box in the transformed perspective also took trial and error. To make the searching process easier, I used a test image that contains lanes that are known to be a straight line and tried different combination of coordinates to get straight lanes in the transformed perspective.

 Here is an example of perspective transform:
<img src="./output_images/perspective_transform.png" />

Sobel Edge Detector
---

The Sobel Edge detector detects edges in images, which is useful for detecting the lanes. I used sobel edge detector to find any sharp edges within specified range of orientations. I had to experiment a lot with the parameters to detect edges with desirable magnitude and orientation:

<img src="./output_images/sobel.png" />

Color Transform
---

Another feature we can make use of for detecting the lanes is their color. Transforming the input image into different colour space allows us to select specific type of colour in more sophisticated manner. I used the following colour filtering pipeline to detect the lanes:

* transforming the input image into Hue Saturation Value (HSV) space and then thresholding in the H and S channels to detect the lanes
* transforming the input image into LUV space and thresholding in the L channel for better detection of white lanes
* transforming the input image into Lab space and thresholding in b space for better detection of yellow lanes

Combined filtering gave the following result:
<img src="./output_images/color.png" />

Polynomial Line Fitting
---

To uncover the shape of the lanes, we first need the starting point of the lanes. This is done by applying histogram method. The histogram method allows us to discover the point in the along width of the images is most likely to be the base of a lane. Once the base of a lane is discovered, sliding windows search method is used to discover rest of the pixels that belong to the lane. The coordinates of those pixels are then used to perform 2nd order polynomial line fitting to model the shape of the line.

Here is the result of polynomial line fitting:
<img src="./output_images/polynomial_fitting.png" />

Entire Pipeline
---
Here are some examples of results for the entire pipeline:
<img src="./output_images/processed_straight_lines1.jpg" />
<img src="./output_images/processed_test3.jpg" />
<img src="./output_images/processed_test5.jpg" />

Smoothing the Line Model Over Time
---
I averaged out model for 10 consecutive frames so that the algorithm is robust against sudden failures to detect lanes or misclassifications of lanes.

## Result

Here is the link to my output video:
[link to my video result](./project_video_output.mp4)

## Discussion
The pipeline was able to detect the lanes in the given project video very well. It did not perfowm as well with the challenge videos. Following is the summary of what I think would improve the algorithm to perform well on the challange videos.

False Positives Rejection Based on Physical Behaviour of the Lanes
---
It is not possible for the lane lines to suddenly change direction or its location. The pipeline tended to produce a lot of false positives when the pavement on the road looked irregular or the lanes get hard to see because of a large shadow. These false positive models can be rejected based on a model of how a lane physically behaves. In these cases, we can make reasonable prediction of where the lanes should based on our model of the lanes.

False Positives Rejection Based on Overall Brightness In the Image
---
Sometimes the camera gets interference from direct sunlight making it hard to detect lanes. These kind of interference can be detected by looking at the overall brightness of the image and see if reliable detection is possible. If not, we can predict where the lanes should be based on historical data about the lanes and use them instead.

