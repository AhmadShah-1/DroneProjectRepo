
import cv2

img = cv2.resize(cv2.imread('Images/testimage2.jpg', 0), (500, 500))
H, W = img.shape

template_height = int(H/5)
template_width = int(W/5)

template = cv2.resize(cv2.imread('Images/template1.png', 0), (template_height, template_width))
h, w = template.shape

# So far cv2.TM_CCOEFF seems the best
methods = [cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR, cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]

for method in methods:
    # Create a new image to try the methods on rather than altering the original image
    img2 = img.copy()

    # Do the template matching
    # Pass the image
    # Pass the template you are looking for
    # Pass the method you are using
    # the .matchTemplate() will create a convulsion, so it will slide the template across the image and see where the highest relation is on the base image
    # Will return a 2d Array showing the accuracy in each region of the image
    result = cv2.matchTemplate(img2, template, method)

    # Store the minimum value
    # Store the maximum value
    # Store the minimum location
    # Store the maximum location
    # Depending on which method you are using, you may need the maximum value or the minimum value for the algorithm
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    # Depending on the method we are using save the min location or the max location
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        location = min_loc
    else:
        location = max_loc

    # Define the bottom right section of the rectangle, this will create rectangle the same size as the template image
    bottom_right = (location[0] + w, location[1] + h)
    cv2.rectangle(img2, location, bottom_right, (128,128,128), 5)
    cv2.imshow('Match', img2)
    cv2.imshow('template', template)
    cv2.waitKey(0)
    cv2.destroyAllWindows()