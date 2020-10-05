# Comparison of two segmentation methods for Lakes Datasets
### Computer vision project on satellite image segmentation 

## I. Setting the task. Pipeline.

The semantic segmentation problem is a rather complex one that is currently being actively addressed. 
Many different techniques are being used, both with complex architecture and with complex training policies. 
A vivid example of a complex learning policy is the transfer learning. Potentially, the situation looks very tempting. 
We have a pre-trained network that does not know nothing about the task we want to solve, 
but already has some preliminary knowledge. Most likely, because we have given this 
preliminary knowledge in the form of the weights of the trained network, it will learn a bit 
better in the end, i.e. it will find the dependencies a bit faster and perhaps a bit more correctly. 



The classical architecture for a semantic segmentation is the convolutional neural network U-Net, this is the first choice to be considered.
(TODO: insert the U-Net architecture pic). The U-Net has an interesting feature - 
if one looks at its encoder part, in the classic version the encoders architecture is exactly the same as that of the VGG neural network. 
The idea that was successfully implemented by Iglovikov in his TernausNet 
(https://arxiv.org/abs/1801.05746), is to train the entire network with a pre-trained VGG encoder. 
Important point of any problem of segmentation is that the labeling, or the creation of a pixel-wise mask for each object - 
in our current problem lake or, in general, water - on the image is a time-consuming and demanding task. VGG is a network 
that has been trained for a classification problem. For the latter it is much easier to assemble a large dataset. 
This is why we are in an interesting situation where we do the transfer learning not just between problems of the same formulation. 
For example, we have a network that was able to classify a thousand classes 
of ImageNet, but we want to distinguish just water from non-water, i.e. to solve a narrower classification 
problem. In this way, we use the classification problem as a basis for the segmentation task. 
We will use VGG11 neural network pretrained on ImageNet. The Dataset that we have is small, and yet it would be 
hard to collect such on our own. In order to assemble it, we used the service of the Yandex Toloka 
platform and set up almost automatic processing of the results that one obtains from there. We got such 
exceptionally high-quality masks (TODO: insert an example of an image and a mask; show the learning curve of Yandex Toloka). 
Thus with someone else's hands we segment the images, and now all we need is a little more money to 
get even larger datasets. With such an automatically set-up pipeline, we have collected 200 
satellite images with lakes and their masks of a large size approximately 9000 by 9000 pixels. 
Next, the task can be solved in different ways: one can, for example, 
resize every image we have to a certain size and try to get a clear mask with the 
network immediately. But we choose a different solution: each image
is cropped into many rectangles, each having a characteristic size approximately such that a 
human being is still able to distinguish water from ground and forest on each crop. 
We have the segmentation masks for each crop accordingly, so we feed them to the network to be trained and, 
voila, the network is able to learn. After that we cover the image with several layers with these 
crops, namely, we take the entire image on a square of 300 by 300 pixels, and then we 
shift each square by another 150 pixels, i.e. we shift the entire grid by a vector (150, 150). 
In total, it turns out that each point is covered by two squares. This makes it possible 
to increase the accuracy of the predictions, because the results that come from these squares 
are averaged. This is the test time augmentation (TTA) method. 

## II. Training the models for the segmentation problem. 
In total, we have crops with sizes from around 300 to around 500 pixels. 
We have to make sure that the network learns how to segment them. 
Since we have decided to use a pre-trained network, this imposes certain limitations. 
The input that we can consider must be exactly the input of the pre-trained network, 
i.e. the characteristic pre-processing to which the images were exposed when they entered ImageNet. 
This means that the input should be standardized to a fixed size 224 x 224 x 3 RGB image, 
and the normalisation should be fixed. This was done, plus custom augmentation was added. 
The most interesting part of an augmentation are the rotations.
So as to guarantee that there are no artifacts from the rotations, we made sure that the rotations were correct.
Namely we only implemented from scratch those rotations that are not to be found in the ready-to-use libraries, 
because the built-in rotations produce artifacts that can affect the learning. 
Since we have a unique situation where there is a huge image from which we cut crops, 
we decided to do the most honest crops possible. 
This is a very specific situation: it is a rare case where one can rotate honestly, preserving the characteristic size 
of the features and not adding any noise due to the artifacts of the rotation of the rectangle. In addition, 
since it is possible to analyse the statistics of the share of water on each crop, 
it turns out that there are too many crops with only land and no water. 
(TODO: insert the generated distribution graph here). In order to make it easier for the network to learn, 
so that it does not fall into overfit, marking all pixels with either earth or water, we balanced the dataset. 
At that we have balanced the already rotated dataset, which guarantees that the training will be more correct (TODO: add explanation). 
After we balance the dataset, we started the training, and this is where the most interesting part is. We can compare two methods - 
training from scratch and training with a pretrained VGG encoder. And, voila, 
it turns out that the model obtained from training the network with an initially pretrained on ImageNet 
(a dataset in principle very distinct from a lakes sattellite image dataset) encoder gives much better results!

TODO: provide a link for the demo with inference.


## III. Results. 
TODO: show Loss function graphs for the two models. Explain, that they show, that the pretrained on ImageNet encoder helps avoid overfitting.
TODO: Provide the final two numbers - the Mean Intersection over Union accuracy metrics (MIOU, or the so-called Dice Loss). 

Technical stack details: 1) everything was implemented from scratch for educational purposes, 
i.e. pytorch was used without frameworks, 
2) custom augmentaion was implemented manually, without usage of the built-in libraries, 3) training was carried out on a vast.ai
