# Advanced Lane Finding Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

This project is part of the [Udacity Self-Driving Car Nanodegree](https://www.udacity.com/drive) program, and some of the code are leveraged from the lecture materials.

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[calibration1]: ./camera_cal/calibration1.jpg "Distorted"
[calibration1_undistort]: ./output_images/calibration1_undistort.jpg "Distortion corrected"
[test1]: ./test_images/test1.jpg "Test 1"
[undistorted]: ./output_images/test1_undistort.jpg "Undistort"
[s_binary]: ./output_images/test1_s_binary.jpg "S Binary"
[x_sobel]: ./output_images/test1_x_sobel_binary.jpg "X Sobel"
[binary_select]: ./output_images/test1_binary_select.jpg "Binary Select"
[masked]: ./output_images/test1_masked_edges.jpg "Masked"
[warped]: ./output_images/test1_binary_warped.jpg "Warped"
[sliding_window]: ./output_images/test1_sliding_window.jpg "Sliding Window"
[skipping_window]: ./output_images/test1_skipping_window.jpg "Skipping Window"
[annotated]: ./output_images/test1_annotated.jpg "Annotated"

## [Rubric Points](https://review.udacity.com/#!/rubrics/571/view)

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is in: https://github.com/fang-yu-liu/advanced-lane-line-detection/blob/master/camera_calibration.py

And this is the Jupyter notebook for visualization: https://github.com/fang-yu-liu/advanced-lane-line-detection/blob/master/camera_calibration_visualization.ipynb

First, going through all the images in "./camera_cal" and preparing "object points" and "image points" for each image. Object points represents the 3d points in the real world space and image points represents 2d points in the image plane where z = 0. Object points are generated using the dimension of the image. Image points are located using the `cv2.findChessboardCorners()` function.

Then, those object points and image points are used to compute the camera calibration matrix and distortion coefficients using the `cv2.calibrateCamera()` function. The calibration result is saved to "./camera_calibration.p" for future use of distortion correction.

Example of applying the distortion correction to an example image using the `cv2.undistort()` function are as follows:

Before calibration | After calibration
------------------ | -----------------
![calibration1][calibration1] | ![calibration1_undistort][calibration1_undistort]

### Pipeline (single images)

I'll use "./test_images/test1.jpg" to demonstrate the pipeline.
![test1][test1]

#### 1. Provide an example of a distortion-corrected image.

`undistort.py`

Original Image  | Distortion-corrected
--------------- | -----------------
![test1][test1] | ![undistorted][undistorted]


#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines 276 through 280 in `Video.py` and details code for thresholding are in `threshold_select.py`).

1. Conver the image to hls domain. Using
2. Color thresholds: Perform a threshold select against s channel. (Threshold = (130, 255))
3. Gradient thresholds: Perform a threshold select for x gradient on l channel
4. Use `cv2.bitwise_or` to obtain the final result of binary select image.

Distorted Image            | S Channel Select      | X Sobel Select | Combined binary select
-------------------------- | -------------------------- | -------------------------- | --------------------------
![undistorted][undistorted]| ![s_binary][s_binary] | ![x_sobel][x_sobel] | ![binary_select][binary_select]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `perspective_transform()`, which appears in `perspective_transform.py`. The function takes an image as the input and use hardcoded source (`src`) and destination (`dst`) points to calculate the perspective transform matrix using `cv2.getPerspectiveTransform` function. And then use the `cv2.warpPerspective` function to obtain the perspective transform for the given image.

```python
src = np.float32([[200, 720],[1100, 720],[580, 455],[700, 455]])
dst = np.float32([[300, 720],[1000, 720],[300, 0],[1000, 0]])
```

The steps in the pipeline for perspective tranform are in line 282 through 287 in `Video.py`.

1. Mask the image for the areas that we want to focus on. (`mask.py`)
2. Perform perspective transform on the masked image. (`perspective_transform.py`)

Masked Image      | Warped
----------------- | -----------------
![masked][masked] | ![warped][warped]


#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

I used the provided code for skipping window and sliding window to fit my lane lines with a 2nd order polynomial. (line 21 through 181 in `Video.py`)

Sliding Window    | Skipping Window
----------------- | -----------------
![sliding_window][sliding_window] | ![skipping_window][skipping_window]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in the `measuring_curvature()` function in `Video.py`. (lines 183 through 211)

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I did this in the `annotate()` function in `Video.py`. (lines 213 through 269)

![annotated][annotated]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](https://github.com/fang-yu-liu/advanced-lane-line-detection/blob/master/output_videos/project_video_output.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I'll describe my general approach of the line detection first and then discuss some potential improvement.

Here is the steps of the pipeline that process each frame in the video: (code block 3 in `advanced_lane_line_video.ipynb` and line 271 through 310 in `Video.py`):

1. Create an Video object with camera calibration matrix and distortion coefficient
2. Process each frame in the video
  1. Distortion correction
  2. Threshold select
  3. Mask
  4. Perspective transform
  5. If reset, perform sliding window search. If not, perform skipping window search.
  6. Measure curvature
  7. Update line information for left line and right line (line 54 through 73 in `Line.py`)
    * Calculate the difference between current fit and previous fit. If the difference is too big, then treat it as a bad fit and increase the `bad_detection` by 1. If the difference is reasonable, then add current fit to recent fits and take average along recent fits (across `n` fits) to get the best fit for lane line.
  8. If `bad_detection` for right/left line is more than `max_bad_detection`, then set reset to True. Clear all recent fits and redo sliding window search.

Potential improvement approaches:
* Fine tune the threshold select to get a better binary select image
* Apply directional or magnitude threshold select to gain more information
* Better mechanism to determine whether the line is a bad detection other than just the difference of the coefficients
* Separating the reset logic for right and left lines instead of resetting them together
