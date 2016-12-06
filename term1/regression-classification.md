# Regression and Classification

Charles and Michael from Georgia Tech

## Classification vs Regression

**Classification**: Taking some input X and map it to some label. Like True or False.
Example: Picture of someone and map if it's male or female.

**Regression**: Continous Value Function
Example: Given some points, map another point

## Classification

### Terms

- Instances: Input, e.g. pixels of picture, credit score, etc.
- Concept: Function, maps input to output
- Target Concept: The answer, the thing we are trying to find
- Hypothesis (Class): Set of all concept we are willing to think about
- Sample: Training Set, Set of all our inputs with a label
- Candidate: Concept that you think might be the Target Concept
- Testing Set: Looks like a training set, determine if candidate concept is doing a good job. Must not be the same as the training set

## Regression

Finding a function to find a relationship between a X and Y label.

### Example with Calculus

#### Finding the best constant Function

`f(x) = c`

Error/Loss function:
$$\sum_{i=1}^n (y_i-c)^2$$

(The sum over all data points of the square difference between the constant and what the value is)

$$0 = -\sum_{i=1}^n 2*(y_i-c)$$

$$n*c = \sum_{i=1}^n y_i$$

**Best constant is mean/average of Y**

## Error

Error can come from different sources. E.g from wrong sensor data, malicious data (bad data given) or by a transcription error.
To cross out any error, we use cross validation.
We can use a test set to "double-check" our predictions. Plus, we can cross validate (taking some parts of the test and the training set) to check even further.

**The goal is always to generalize**
