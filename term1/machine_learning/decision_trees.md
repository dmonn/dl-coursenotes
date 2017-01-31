# Decision Trees

Decision Trees allow you to ask multiple linear questions (Is it windy? Is it sunny?)


## Coding

```
from sklearn import tree

def classify(features_train, labels_train):

    ### your code goes here--should return a trained decision tree classifer
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(features_train, labels_train)

    return clf
```

To calculate the accuracy:

```
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features_train, labels_train)
right = []

for i,test in enumerate(features_test):
    prediction = clf.predict(np.array(test).reshape(1, -1))
    if prediction == labels_test[i]:
        right.append(True)


print(len(right), len(labels_test))
acc = float(len(right)) / float(len(labels_test)) * 100
```

Very prone to overfitting! (Tries to fit all the data) with the default parameters.

The parameter `min_sample_split` stands for the minimum samples it need to still split the data (edge-cases). Default number is 2. Maybe try 50.

## Entropy

Measure of impurity in some examples.

The formula is `sum of -pi*log2(pi)`. P are all the points in class i.

### Information Gain

`information gain = entropy(parent) - (weighted average) of entropy(children)`

Target: Maximize information gain, deicde where to split

Weighted average = entropy of child * percentage of whole  (e.g. if there are 5 examples, 2 are x, then weighted average of x) = 2/5

## Criterion

Sets the criteria of split, either 'gini' or the above 'entropy'.


## Bias - Variance

High Bias: Ignoring the data
High Variance: Can't generalize, only uses given data

## Pros and Cons

+ Easy to use and explain
+ Graphically showable (perfect for demonstration)
+ Work offline
+ Easy to understand

- Prone to overfitting
- Not good for big data
