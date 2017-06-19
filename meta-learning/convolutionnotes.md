# Convolutions

## Formula

`Output = (Input - Filter + 2*Padding)/Stride + 1`

## Basic Example

Filter / Kernel: 5 x 5 (x 3)
Input Img: 32 x 32 x 3
Output: 28 x 28

## Stride and Padding

The amount by which the filter shifts is the stride. (1 in example above)

Filter / Kernel: 3 x 3
Input Img: 7 x 7
Stride: 1
Output Img: 5 x 5

Filter / Kernel: 3 x 3
Input Img: 7 x 7
Stride: 2
Output Img: 3 x 3

Padding applies a "frame" around the image

## Pooling Layers

In this category, there are also several layer options, with maxpooling being the most popular. This basically takes a filter (normally of size 2x2) and a stride of the same length. It then applies it to the input volume and outputs the maximum number in every subregion that the filter convolves around.

## TF Specific: Valid vs. Same

```
"VALID" = without padding:
   inputs:         1  2  3  4  5  6  7  8  9  10 11 (12 13)
                  |________________|                dropped
                                 |_________________|
"SAME" = with zero padding:
               pad|                                      |pad
   inputs:      0 |1  2  3  4  5  6  7  8  9  10 11 12 13|0  0
               |________________|
                              |_________________|
                                             |________________|
```
