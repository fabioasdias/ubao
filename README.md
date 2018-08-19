# ubao

- We have a "training board", 20 images positioned in a board - Ys on the code.
(https://github.com/fabioasdias/ubao/blob/master/train.svg)

- We have a pre-trained neural network (VGG19 - https://arxiv.org/abs/1409.1556) that we use to generate feature vectors. This feature vector "summarizes" the image using an n-dimensional vector (I don't remember the value of n). That's Xs on the code.

This pair (Xs, Ys) defines a transformation from Rn to R2. We then use the Local Affine Multidimensional Projection (LAMP - http://www.lcad.icmc.usp.br/~nonato/pubs/lamp.pdf ) to replicate this same transformation for the new images, so images with similar feature vectors get placed near each other on the board.

This whole thing gets saved into a .json file that is used by the interface to show the images (the folder 'ui').

The "magic" of the thing lies in the VGG network, which comes pre-trained to perform object recognition. We use most of the network, but we cut off the last layer (the one that actually tells you which objects are on the image) and use the output of the previous layer to define the feature vector. It is the same principle used for natural language processing to define feature vectors for words.
