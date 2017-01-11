# Lane Line Detection

In our first project, we learn how to select lane lines.

## Color Selection

A picture with lane lines has three colors: Red, Green and Blue.
Each value can be range from 0 (darkest) to 255 (brightest).
So white would be 255 for each value.

### Code sample to black out everything

Check out the code below. First, I import pyplot and image from matplotlib. I also import numpy for operating on the image.

```
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
```

I then read in an image and print out some stats. I’ll grab the x and y sizes and make a copy of the image to work with.

```
# Read in the image and print out some stats
image = mpimg.imread('test.jpg')
print('This image is: ',type(image),
         'with dimensions:', image.shape)

# Grab the x and y size and make a copy of the image
ysize = image.shape[0]
xsize = image.shape[1]
color_select = np.copy(image)
```

Next I define a color threshold in the variables red_threshold, green_threshold, and blue_threshold and populate rgb_threshold with these values. This vector contains the minimum values for red, green, and blue (R,G,B) that I will allow in my selection.


```
# Define our color selection criteria
red_threshold = 0
green_threshold = 0
blue_threshold = 0
rgb_threshold = [red_threshold, green_threshold, blue_threshold]
```

Next, I'll use a bitwise OR to select any pixels below the threshold and set them to zero.

After that, all pixels that meet my color criterion will be retained, and those that do not will be blacked out.


```
# Use a "bitwise OR" to identify pixels below the threshold
thresholds = (image[:,:,0] < rgb_threshold[0]) \
            | (image[:,:,1] < rgb_threshold[1]) \
            | (image[:,:,2] < rgb_threshold[2])
color_select[thresholds] = [0,0,0]

# Display the image                 
plt.imshow(color_select)
```

The result, color_select, is an image in which pixels that were above the threshold have been retained, and pixels below the threshold have been blacked out.

In the code snippet above, red_threshold, green_threshold and blue_threshold are all set to 0, which implies all pixels will be included in the selection.

### Region of interest

Now, if we black out everything, there are still some random pixels.
We don't want this. That's why we cut out a triangle which is our "search-area."

However, sometimes lane lines could also be yellow or they could have a different lightning.
How are we dealing with this?


## Computer Vision

### Canny Edge Detection

Canny Edge Detection is a algorithm which helps us to detect edges.

First, we need to read in an image:

```
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
image = mpimg.imread('exit_ramp.jpg')
plt.imshow(image)
```

Let's go ahead and convert to grayscale.

```
import cv2  #bringing in OpenCV libraries
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) #grayscale conversion
plt.imshow(gray, cmap='gray')
```

Let’s try our Canny edge detector on this image. This is where OpenCV gets useful. First, we'll have a look at the parameters for the OpenCV Canny function. You will call it like this:

```
edges = cv2.Canny(gray, low_threshold, high_threshold)
```

The algorithm will first detect strong edge (strong gradient) pixels above the high_threshold, and reject pixels below the low_threshold.

The low to high threshold ratio should be about 1:2 to 1:3.

We are also doing a GaussianBlur to reduce the noise.

```
#doing all the relevant imports
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

# Read in the image and convert to grayscale
image = mpimg.imread('mountain_road.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# Define a kernel size for Gaussian smoothing / blurring
kernel_size = 3
blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size), 0)

# Define parameters for Canny and run it
# NOTE: if you try running this code you might want to change these!
low_threshold = 1
high_threshold = 10
edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

# Display the image
plt.imshow(edges, cmap='Greys_r')
```

### Using the Hough Transform to Find Lines from Canny Edges

In image space, a line is plotted as x vs. y, but in 1962, Paul Hough devised a method for representing lines in parameter space, which we will call “Hough space” in his honor.

In Hough space, I can represent my "x vs. y" line as a point in "m vs. b" instead. The Hough Transform is just the conversion from image space to Hough space. So, the characterization of a line in image space will be a single point at the position (m, b) in Hough space.
