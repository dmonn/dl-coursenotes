# SVM (Support Vector Machines)

Outputs a line to seperate data


## Margins

To seperate the data, it maximizes the distance to the nearest class/point (margin).
This can lead to errors if some of the data is tricky.
First Priority is always classifying correctly, second is maximizing the margin.


## Outliers

Sometimes a data-point can't be classified correctly. In this case, it tries to do the best it can.
Therefore it ignores outliers

## Non-Linear SVM

Given a new feature, you can create non-linear SVM.


### Parameters


You can use `scikit-learn`'s kernel functions to solve non-linear svm's. For example `sklearn.svm.SVC` (Classifier).
You can also use different kernels, such as 'linear' or 'poly'.

Other parameters are `C` and `Gamma`.

The `C`-parameter controls between smooth decisions and classifying correctly.
A higher `C` means more training points are correctly.

The `Gamma` parameter defines the influence of a single training example. Low value = far reach.

## Pros and Cons

+ Work well with smaller dataset
+ Flexible, a lot of parameters 

- Does not work well with big datasets
- Does not work well with a lot of noise, overlapping data
- Slow and prone to overfitting
