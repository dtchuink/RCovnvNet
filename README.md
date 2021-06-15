# RCovnvNet

A simple Recurrent convolution neural network for 3D object recognition

The number of points required to represent a bed is not the same needed to repre-sent a cup. Thus, if the bed has the same number of pointas the cup it might not be recognized.  

3D object recognition systems typi-cally use RNNs combine to a CNN to learn higher order fea-tures. In this case, the RNN can be seen as combining convolution and pooling into one efficient operation. Our approachis different in that we use RNN to feed the model with in-puts of different sizes. 




## Reference
[1] Tchuinkou, Danielle, and Christophe Bobda. "R-covnet: Recurrent neural convolution network for 3d object recognition." 2018 25th IEEE International Conference on Image Processing (ICIP). IEEE, 2018.
