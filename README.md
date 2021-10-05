# Deep-Gaze-2
Deep Gaze II Tensorflow implementation 

Original Paper:

DeepGaze II: Reading fixations from deep features trained on object recognition 
Author Matthias Kümmerer et. al. [https://arxiv.org/abs/1610.01563]

### The archetecture

![Network Archetecture](https://github.com/Po-Hsuan-Huang/Deep-Gaze-2/blob/main/ReadMe_imgs/arche.png)

The network consists of a VGG16 feature extractor, Feature Readout Module, and output classification softmax layer.

The feature readout module recieve outputs from conv5_1, relu5_1, relu5_2, conv5_3, relu5_4. However, the Keras VGG16 pretrained network doesn't allow access to conv5_1, conv5_3 as mentioned in the paper, we chose conv4_3, conv4_pool, relu5_1, relu5_2, relu5_3 alternatively.  

The weights of VGG16 model are fixed in the original paper in order to save time training. In our model, we allow training of weights in conv4_1, cvon4_2, conv4_3, pool4, conv5_1, conv5_2, conv5_3, pool5 to improve prediction accuracy.

Instead of using L1 regulation, we use L2 regularion in the Readout Module.

Unlike the original paper where the size and the variance of gaussian kernel are also trainable parameters. They are fixed variables in our model. User has to optimize them manually. 

Center bias of the training and validation set are preprocessed into probability and saved as numpy arrays. The center bias then is loaded as aux input during training.

### Training

The paper pretrained the readout module with SALICON dataset with 10 fold cross-validation, then fine-tune it with MIT300 dataset to prevent overfitting. 

In this implementation we pretrained the network with SALICON dataset

### Evaluation

In this implemetnation we only evaluate with sAUC, Judd-AUC, Borji-AUC, NSS, CC, and IG. The evaluation metrics are from other source,and they are #not verified#.


