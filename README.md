# Road-Pixel-Semantic-Segmentation
KITTI Road Semantic Segmentation Dataset
Problem: Predict the Road pixels in a image

# Dataset:
Based on KITTI Road semantic segmentation dataset. We are provided with 289 images, their ground truth images and labels( as binary files, each file stores a (352,1216) numpy array representing the per-pixel numerical label of a image). Each image is 352 x 1216 RGB image.In this dataset, 0 for Non-Road , 1 for Road and -1 for Void.

# Architecture:
A Fully Convolutional Network FCN-32s. 

FCN-32s network is based on VGG-16 network with 3 modifications:

1. Replace the first fc layer by a convolutional layer with kernel size of 7. The
number of the kernels is the output dimension of the original fc layer, which is
4096 ;
2. Replace the second and third fc layers by convolutional layers with kernel size of
1. The numbers of kernels are 4096 and 1 respectively.
3. Add a deconvolutional layer after the third fc layer.

# Evaluation Metric:
Pixel-level IoU = TP/(TP+FP+FN),  (IoU= Intersection over Union)
where TP, FP, and FN are the numbers of true positive, false positive, and false negative pixels, respectively

# Training:
The training was performed for 20 epochs using a batch size of 1(one image already includes large number of pixel samples), SGD optimizer with momentum(tf.train.MomentumOptimizer) and learning rate = 0.001, momentum = 0.99.

# Results:
Find the trends of the Loss and IOU for the Training and Validation data in the curves folder.

Mapped the output predicted labels to color images which looked like the ground truth images.  
(Note: Black pixels in the actual image are just noise for us as our primary aim of the project is to segment the Road region
hence, black pixels are not visible in the predicted image.)

# OUTPUT by the program and ACTUAL Ground truth images:

