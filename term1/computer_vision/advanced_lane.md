# Advanced Lane Finding

## Distortion

Image distortion means looking at 3D objects in a 2D image. This can lead to wrong objects in the world.

In math, this transformation from P(X, Y, Z) to p(x, y) is done by the Camera Matrix (P ~ Cp).

There is *Radial Distortion* which makes straight lines seems bent.

There is *Tangental Distortion* which makes the image looks tilted.

There are three coefficients needed to correct for radial distortion: k1, k2, and k3. To correct the appearance of radially distorted points in an image, one can use a correction formula.

There are two more coefficients that account for tangential distortion: p1 and p2, and this distortion can be corrected using a different correction formula.

## Practical Usage (Chessboard)

In this exercise, you'll use the OpenCV functions `findChessboardCorners()`and `drawChessboardCorners()` to automatically find and draw corners in your image.


```
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# prepare object points
nx = 8#TODO: enter the number of inside corners in x
ny = 6#TODO: enter the number of inside corners in y

# Make a list of calibration images
fname = 'calibration_test.png'
img = cv2.imread(fname)

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Find the chessboard corners
ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

# If found, draw corners
if ret == True:
    # Draw and display the corners
    cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
    plt.imshow(img)

```


## Calibrating a Camera

To calibrate the camera, one can add the object points and image points in an array and then call the following function:

```
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[::-1],None,None)
```

Then undistort the image:

```
dst = cv2.undistort(img, mtx, dist, None, mtx)
```

## Lane Curvature

Self-driving cars need to be told the correct steering angle to turn, left or right. You can calculate this angle if you know a few things about the speed and dynamics of the car and how much the lane is curving.

For a lane line that is close to vertical, you can fit a line using this formula: f(y) = Ay^2 + By + C, where A, B, and C are coefficients.

## Perspective Transform

A perspective transform maps the points in a given image to different, desired, image points with a new perspective. The perspective transform you’ll be most interested in is a bird’s-eye view transform that let’s us view a lane from above; this will be useful for calculating the lane curvature later on. Aside from creating a bird’s eye view representation of an image, a perspective transform can also be used for all kinds of different view points.

Compute the perspective transform, M, given source and destination points:

```
M = cv2.getPerspectiveTransform(src, dst)
```

Compute the inverse perspective transform:

```
Minv = cv2.getPerspectiveTransform(dst, src)
```

Warp an image using the perspective transform, M:

```
warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
```

## Quiz

Undistort and Transform

```
Define a function that takes an image, number of x and y points,
# camera matrix and distortion coefficients
def corners_unwarp(img, nx, ny, mtx, dist):
   # Use the OpenCV undistort() function to remove distortion
   undist = cv2.undistort(img, mtx, dist, None, mtx)
   # Convert undistorted image to grayscale
   gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)
   # Search for corners in the grayscaled image
   ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

   if ret == True:
       # If we found corners, draw them! (just for fun)
       cv2.drawChessboardCorners(undist, (nx, ny), corners, ret)
       # Choose offset from image corners to plot detected corners
       # This should be chosen to present the result at the proper aspect ratio
       # My choice of 100 pixels is not exact, but close enough for our purpose here
       offset = 100 # offset for dst points
       # Grab the image shape
       img_size = (gray.shape[1], gray.shape[0])

       # For source points I'm grabbing the outer four detected corners
       src = np.float32([corners[0], corners[nx-1], corners[-1], corners[-nx]])
       # For destination points, I'm arbitrarily choosing some points to be
       # a nice fit for displaying our warped result
       # again, not exact, but close enough for our purposes
       dst = np.float32([[offset, offset], [img_size[0]-offset, offset],
                                    [img_size[0]-offset, img_size[1]-offset],
                                    [offset, img_size[1]-offset]])
       # Given src and dst points, calculate the perspective transform matrix
       M = cv2.getPerspectiveTransform(src, dst)
       # Warp the image using OpenCV warpPerspective()
       warped = cv2.warpPerspective(undist, M, img_size)

   # Return the resulting image and matrix
   return warped, M
```

## Sobel Operator

You need to pass a single color channel to the cv2.Sobel() function, so first convert to grayscale:

```
gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
```

Note: Make sure you use the correct grayscale conversion depending on how you've read in your images. Use cv2.COLOR_RGB2GRAY if you've read in an image using mpimg.imread() or cv2.COLOR_BGR2GRAY if you've read in an image using cv2.imread().

Calculate the derivative in the x-direction (the 1, 0 at the end denotes x-direction):

```
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
```

Calculate the derivative in the y-direction (the 0, 1 at the end denotes y-direction):

```
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
```

Calculate the absolute value of the x-derivative:

```
abs_sobelx = np.absolute(sobelx)
```

Convert the absolute value image to 8-bit:

```
scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
```

Create a binary threshold to select pixels based on gradient strength:

```
thresh_min = 20
thresh_max = 100
sxbinary = np.zeros_like(scaled_sobel)
sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
plt.imshow(sxbinary, cmap='gray')
```

## Color Spaces

When we grayscale the image, we can loose valuable information.

RGB is red, green and blue.
There is also HSV color space (hue, saturation, and value), and HLS space (hue, lightness, and saturation). These are some of the most commonly used color spaces in image analysis.

```
# Define a function that thresholds the S-channel of HLS
def hls_select(img, thresh=(0, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
    return binary_output
```

## Using Both

```
# Convert to HLS color space and separate the S channel
# Note: img is the undistorted image
hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
s_channel = hls[:,:,2]

# Grayscale image
# NOTE: we already saw that standard grayscaling lost color information for the lane lines
# Explore gradients in other colors spaces / color channels to see what might work better
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# Sobel x
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0) # Take the derivative in x
abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

# Threshold x gradient
thresh_min = 20
thresh_max = 100
sxbinary = np.zeros_like(scaled_sobel)
sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

# Threshold color channel
s_thresh_min = 170
s_thresh_max = 255
s_binary = np.zeros_like(s_channel)
s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1

# Stack each channel to view their individual contributions in green and blue respectively
# This returns a stack of the two binary images, whose components you can see as different colors
color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary))

# Combine the two binary thresholds
combined_binary = np.zeros_like(sxbinary)
combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1

# Plotting thresholded images
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
ax1.set_title('Stacked thresholds')
ax1.imshow(color_binary)

ax2.set_title('Combined S channel and gradient thresholds')
ax2.imshow(combined_binary, cmap='gray')
```

## Lane Curvature

```
import numpy as np
# Generate some fake data to represent lane-line pixels
yvals = np.linspace(0, 100, num=101)*7.2  # to cover same y-range as image
leftx = np.array([200 + (elem**2)*4e-4 + np.random.randint(-50, high=51)
                              for idx, elem in enumerate(yvals)])
leftx = leftx[::-1]  # Reverse to match top-to-bottom in y
rightx = np.array([900 + (elem**2)*4e-4 + np.random.randint(-50, high=51)
                                for idx, elem in enumerate(yvals)])
rightx = rightx[::-1]  # Reverse to match top-to-bottom in y

# Fit a second order polynomial to each fake lane line
left_fit = np.polyfit(yvals, leftx, 2)
left_fitx = left_fit[0]*yvals**2 + left_fit[1]*yvals + left_fit[2]
right_fit = np.polyfit(yvals, rightx, 2)
right_fitx = right_fit[0]*yvals**2 + right_fit[1]*yvals + right_fit[2]

# Plot up the fake data
plt.plot(leftx, yvals, 'o', color='red')
plt.plot(rightx, yvals, 'o', color='blue')
plt.xlim(0, 1280)
plt.ylim(0, 720)
plt.plot(left_fitx, yvals, color='green', linewidth=3)
plt.plot(right_fitx, yvals, color='green', linewidth=3)
plt.gca().invert_yaxis() # to visualize as we do the images
```

## Creating a weighted image

```
# Create an image to draw the lines on
warp_zero = np.zeros_like(warped).astype(np.uint8)
color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

# Recast the x and y points into usable format for cv2.fillPoly()
pts_left = np.array([np.transpose(np.vstack([left_fitx, yvals]))])
pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, yvals])))])
pts = np.hstack((pts_left, pts_right))

# Draw the lane onto the warped blank image
cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

# Warp the blank back to original image space using inverse perspective matrix (Minv)
newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0]))
# Combine the result with the original image
result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
plt.imshow(result)
```
