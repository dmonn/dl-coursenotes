# Tensorflow

Vincent Vanhoucke

## Getting started

"Variables" in Tensorflow are stored in `tensorflow.constant()`

You use a Session to compute the outputs

To have a variable with a later value, you use `tensorflow.placeholder()`

```
x = tf.placeholder(tf.string)

with tf.Session() as sess:
    output = sess.run(x, feed_dict={x: 'Hello World'})
```
or

```
x = tf.placeholder(tf.string)
y = tf.placeholder(tf.int32)
z = tf.placeholder(tf.float32)

with tf.Session() as sess:
    output = sess.run(x, feed_dict={x: 'Test String', y: 123, z: 45.67})
```

The `tf.Variable()` function creates a tensor with an initial value that can be modified

```
init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)

```

```
n_features = 120
n_labels = 5
weights = tf.Variable(tf.truncated_normal((n_features, n_labels)))
```

The `tf.truncated_normal()` function returns a tensor with random values from a normal distribution whose magnitude is no more than 2 standard deviations from the mean.

Since the weights are already helping prevent the model from getting stuck, you don't need to randomize the bias. Let's use the simplest solution, setting the bias to 0.

```
n_labels = 5
bias = tf.Variable(tf.zeros(n_labels))
```

The `tf.zeros()` function returns a tensor with all zeros.
