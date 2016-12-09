# Convolutional Neural Networks

Sometimes you want your network to "remember" something learned before. E.g. how a kitten looks like, even though it's in a different position. You achieve this with "weight sharing". (Statistical Invariants)

## Terms

* Input: E.g a picture
* Patch/Kernel: Small patch of a picture - a small NN
* Feature Map: (R, G, B) for example
* Stride: Amount of pixels you are shifting
  * Stride of 1 is roughly the same of the input, 2 is about half the size

## Dimensionality

Given our input layer has a volume of W, our filter has a volume (height * width * depth) of F, we have a stride of S, and a padding of P, the following formula gives us the volume of the next layer: (W−F+2P)/S+1.

```
new_height = (input_height - filter_height + 2 * P)/S + 1
new_width = (input_width - filter_width + 2 * P)/S + 1
```

**Remember the depth is equal to the number of filters!**

In code:

```
input = tf.placeholder(tf.float32, (None, 32, 32, 3))
filter_weights = tf.Variable(tf.truncated_normal((8, 8, 3, 20))) # (height, width, input_depth, output_depth)
filter_bias = tf.Variable(tf.zeros(20))
strides = [1, 2, 2, 1] # (batch, height, width, depth)
padding = 'VALID'
conv = tf.nn.conv2d(input, filter_weights, strides, padding) + filter_bias
```
