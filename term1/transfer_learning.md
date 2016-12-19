# Transfer Learning

Transfer learning is a method where you take existing and pre-trained neural networks and adjust them for your own needs.

In this chapter we took the network `AlexNet` and adjusted it so we could use it for our traffic sign classification.

## VGG

The Visual Geometry Group created a net called VGG-Net. It is a sequence of 3x3 Convolutional Layers, splitted up by 2x2 Pooling Layers and finished by 3 fully connected layers. It's a starting point for many image classification tasks thanks to it's flexibility.

## GoogLeNet

The advantage of this net is, that it runs really fast. They created an inception module.

### Inception

The idea is that on each layer of your ConvNet, you can make a choice (Pooling, Convolution - 5x5, 3x3, 1x1). But instead of choosing, you make them all.

## ResNet

ResNet is from Microsoft. ResNet has a total of 152 (!) layers, while AlexNet has 8, VGG has 19, and GoogLeNet has 21.
It has the smallest error.

## Summary

## Feature Extraction

If dataset is small and similar to the original dataset. The higher-level features learned from the original dataset should be relevant to the new dataset.

## Finetuning

If the dataset is large and similar to the original dataset. In this case we should be much more confident we won't overfit so it should be safe to alter the original weights.

If the dataset is small and very different from the original dataset. You could also make the case for training from scratch. If we choose to finetune it might be a good idea to only use features found earlier on in the network, features found later might be too dataset specific.

## From Scratch

If the dataset is large and very different from the original dataset. In this case we have enough data to confidently train from scratch. However, even in this case it might be more beneficial to finetune and the entire network from pretrained weights.

Most importantly, keep in mind for a lot of problems you won't need an architecture as complicated and powerful as VGG, Inception, or ResNet. These architectures were made for the task of classifying thousands of complex classes. A much smaller network might be a much better fit for your problem, especially if you can comfortably train it on moderate hardware.
