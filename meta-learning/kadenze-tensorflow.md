# Creative Applications of Deep Learning with TensorFlow by Kadenze

These are some notes based on the TensorFlow course by Kadenze.
I only made the first chapter where one could learn how to create Basic Neural Networks with TF.

## Table of Contents

* [Error / Cost Function](#error---cost-function)
	+ [Minimizing](#minimizing)
* [Local Minima/Optima](#local-minima-optima)
	+ [Learning Rate](#learning-rate)
* [Creating a Neural Network](#creating-a-neural-network)
	+ [Setup the data](#setup-the-data)
	+ [Creating the Network](#creating-the-network)
	+ [Training the Network](#training-the-network)
  		- [Mini Batches & Stochastic](#mini-batches---stochastic)
  		- [Non-Linear Functions](#non-linear-functions)




## Error / Cost Function

The error indicates whether or not a neural network predicted something correctly. So if an orange is a "1" and an apple is a "0". We can say the following:

If we have an apple (0) and the NN predicts 0, the error is 0. But if the NN predicts 1, the error is 1.

If we have an orange (1) and the NN predicts 0, the error is 1. If the NN predicts 1, the error is 0.

We calculate that like this:


```
error = prediction(value) - label
E = F(x) - Y
```


### Minimizing

We might have other values for errors, say 100 is a full error. Perhaps we can get e.g. a 50 (partial) error.  Then we can alter our parameters to see if the error gets smaller or bigger. So if we had an error of 50 before, bbut now we have 45, we are on the right track to minimizing our error.

**Cost/Loss are other words for 'Error'**


## Local Minima/Optima

Depending on where our "random" initialisation began, we could get stuck on a "local minima". Which means that our Error will be the lowest in that local point, but it could be smaller on another point of our "Error Graph".

We can solve that problem with the **Learning Rate.**

### Learning Rate


If the learning rate is **too small**, we won't get anywhere.

If the learning rate is **too big**, we might overshoot the minima.


## Creating a Neural Network

### Setup the data

First, we need to create some dummy data. Let's say we have 1000 observations:

```
n_obervations = 1000
```

The input to our network are going to be numbers from -3 to 3.

```
xs = np.linspace(-3, 3, n_observations)
```

We are creating a sine-wave (Sinus), but we are adding some noise (one could say wrong data) to it. The cause of our NN is, to find out the Sine-Wave without any noise.

```
ys = np.sin(xs) + np.random.uniform(-0.5, 0.5, n_observations)
```

We we plot it, there are some dots wo could form some kind of a Sine-Wave.

```
plt.scatter(xs, ys, alpha=0.15, marker="+")
```

### Creating the Network

First, we create a placeholder which we are going to fill in later:

```
X = tf.placeholder(tf.float32, name="X")
```

This is going to specify all of our X-Axis-Values. We do the same thing for the Y-Axis.

```
Y = tf.placeholder(tf.float32, name="Y")
```

Now, we need to set our initial values. We are choosing values close to 0 here.

```
sess = tf.InteractiveSession()
> n = tf.random_normal([1000]).eval()
> plt.hist(n)
```
Now we can see that we have some random values which are all based around 0. But they are still between -3 and 3. We need values closer to 0. We use the Standard Deviation for that.

```
sess = tf.InteractiveSession()
> n = tf.random_normal([1000], stddev=0.1).eval()
> plt.hist(n)
```
Next up, we are going to set a **TensorFlow Variable** to store the initial parameter value.

```
W = tf.Variable(tf.random_normal([1], dtype=tf.float32, stddev=0.1), name="weight")
```
Also, we need a "Bias" Variable which allows us to move the values.

```
B = tf.Variable(tf.constant([1], dtype=tf.float32), name="bias")
```

Now we can scale our input value by Y and add our Bias B to make a prediction.

```
Y_pred = X * W + B
```

**Now we are using the Gradient Descent to figure out what the perfect values for W and B are**

We are trying to transform a value X, which is ranging from -3 to 3 to match the value Y. This value Y should look like a Sine-Wave, which ranges from -1 to 1.

We know how a sine-wave should look like, so we can measure the distance from an output value to a regular sine wave with a easy python function:

```
def distance(p1, p2):
	return tf.abs(p1 - p2)
```

Now we can take our predicted value and our known Sinus-Value of an input and measure the distance, **or the cost/error.**

```
cost = distance(Y_pred, tf.sin(X))
```

If we would have a more complex data, we wouldn't know what the data would look like. So our second parameter ```tf.sin(X)``` would be unknown.

So we would take a set of **known** data to calculate the cost function. This would look something like:

```
cost = distance(Y_pred, Y)
```

### Training the Network

Now we are using a simple Tensorflow Optimizer to train the Network:

```
cost = tf.reduce_mean(distance(Y_pred, Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)
```
There, we tell the optimizer to minimize our cost variable.

The optimizer handles updating all the variables.

```
n_iterations = 500 # We are going to run this 500 times
fig, ax = plt.subplots(1, 1) # Plotting the distribution over time
ax.scatter(xs, ys, alpha=0.15, marker="+") # Predictions
```

When we plot the data, it doens't look like a Sine-Wave. It's just a linear line with the right trend without curves. The "wave" is probably caught in a local minima.

#### Mini Batches & Stochastic

If you have a large set of data, you might want to use Mini Batches. You take a subset of the data to train the network so you have a more generalised picture.

So, let's iterate through our dataset again, one batch at a time.

```
idxs = np.arange(100)
batch_size = 10
n_batches = len(idxs) // batch_size
for batch_i in range(n_batches):
	print(idxs[batch_i * batch_size : (batch_i + 1) * batch_size])
```
There is just one problem: Our batches are still in an order. And Neural Network will pick up any given order. So let's randomise.

```
idxs = np.arange(100)
rand_idxs = np.random.permutation(idxs)
batch_size = 10
n_batches = len(rand_idxs) // batch_size
for batch_i in range(n_batches):
	print(idxs[batch_i * batch_size : (batch_i + 1) * batch_size])
```

#### Non-Linear Functions

Most complex networks aren't linear. That's why we need a Non-Linear function. We can use the same variables as before, but instead of multiplying them, we are going to put them through a non-linearity.

```
h = tf.nn.tanh(tf.matmul(tf.expand_dims(x, 1), W) + b, name='h')
Y_pred = tf.reduce_sum(h, 1)
```

After that, we are going to train in the exact same way.
The previously defined formula, now looks like this:

```
H = nonlinearity(XW + b)
```

