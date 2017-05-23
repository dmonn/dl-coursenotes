# Sentiment Analysis

Our goal in this chapter is to take a human-generated text and predict, whether or not it's positive or negative.
To do this, we need to find patterns in the text, since a Neural Network only takes numbers as input.

## Theory Validation

Our theory is, that there are "good" and "bad" words. Therefore we are calculating the most common words per label.

The solution for this is in the `Sentiment Classification - Project 1 Solution.ipynb` file.

Basically, we take the most common words, then calculate the ratio between the words being good and bad. Also, we need to set a threshold of how many times the word has to be there.

## Transforming Text into Numbers  

First, we are going to count any vocabulary in a review to see, how many times the word exists.
(This is not good and adds noise, set any existing vocab to 1!)
Therefore we are giving every vocabulary a index.
Plus, we are going to set labels 0 for negative, 1 for positive.

## Building the Neural Network

Here we can use the network from last chapter.

## Neural Noise

In the example above, the training wasnt really good. In a situation like this, one should check the hyperparams but the problem is most likely noise.
Deep Learning has the principle "garbage in, garbage out" so always check that your data is good!

In this example, we reduce noise by removing the word count and replace it by just setting 1 to existing vocabulary.

## Understanding Inefficiencies

This is a step to make the learning faster.
1. Vectors shouldn't be too large
2. Don't make useless calculations (e.g 0 times, 1 times)

## More Noise Reduction

In this example, we could remove the unnecessary words and items such as dots, and, or etc.

Here we are taking the previously computed pos/neg-ratio and drop everything below a threshold. Meaning if e.g. there is only 1% difference between the word being positive or negative, we drop it
