# **Behavioral Cloning** 

## Writeup Template


---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


## Rubric Points

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
$ python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed
* My CNN starts with a normalization layer (Line 66)
* Followed by three convolutional layers activated by RELU with max pooling. (Line 68 - 73)
* Ends with several fully connected layers. (Line 74 - 79)


#### 2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

Correction term could be tuned for left and right camera views.

#### 4. Appropriate training data
CNN was trained with normalized image data of center, left and right view. 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

First I used LeNet for training data. I started with center data only. The vehicle doesn't know when to turn.

Then I add code to augment data gradually. 
* Use left and right views to train with adjustment on the steering value.
* Flipped all images.
* Cropped all images to focus on the center portion.

A train/validation paradigm is applied the entire time.

Later for better result, I modified the network published by nvidia suggested by the instructor in video #14.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 66 - 79). Details have been describe previously.


#### 3. Creation of the Training Set & Training Process
The training data is recorded by me driving in the simulator.

To process the data, I first preprocessed the csv file using a script([log_file_preprocess.py](log_file_preprocess.py)). This
script reads the driving log csv file using pandas and returns data info of center, left, right and flip/no-flip for further
process inside the generator.

The image data are flipped when needed and steering data is corrected accordingly for left and right view.

Then batch by batch, training are fed into the CNN describe above.
