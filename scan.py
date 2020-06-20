from pyimagesearch.transform import four_point_transform
from skimage.filters import threshold_local 
import numpy as np
import argparse, cv2, imutils
#____________________________________________________________________________________________________________________________________________

# construct the argument parser and parse the arguments (image path)
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image to be scanned")
ap.add_argument("-v", "--verbose", default=0, help="Show steps") # 0 = Disabled | 1 = Enabled
args = vars(ap.parse_args())
img_path = args['image']
verbose = args['verbose']
#____________________________________________________________________________________________________________________________________________

image = cv2.imread(img_path)                # load the image
ratio = image.shape[0] / 500.0              # get the ratio of the old height
orig = image.copy()                         # clone it
image = imutils.resize(image, height=500)   # resize it

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # convert the image to grayscale
gray = cv2.GaussianBlur(gray, (5,5), 0)         # blur the image
edged = cv2.Canny(gray, 75, 200)                # find the edges

if verbose:
    print("STEP 1: Edge Detection")
    cv2.imshow("Image", image)
    cv2.imshow("Edged", edged)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
#____________________________________________________________________________________________________________________________________________

# get the contours
cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) 
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5] # keep only the large ones

# loop over the contours
for c in cnts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02*peri, True)   # approximate the contour


    if len(approx) == 4:    # if the contour has 4 points, it can be assumed that the screen was found
        screenCnt = approx
        break

if verbose:
    print("STEP 2: Find contours of paper")
    cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
    cv2.imshow("Outline", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
#____________________________________________________________________________________________________________________________________________

warped = four_point_transform(orig, screenCnt.reshape(4, 2)*ratio) # apply the transform to get a top-down view of the original image

# make the black and white look of scanned documents
warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
T = threshold_local(warped, 11, offset=10, method="gaussian")
warped = (warped > T).astype("uint8") * 255

if verbose:
    print("STEP 3: Apply perspective transform")
    cv2.imshow("Original", imutils.resize(orig, height = 650))
    cv2.imshow("Scanned", imutils.resize(warped, height = 650))
    cv2.imshow("Warped", warped)
    cv2.waitKey(0)
#____________________________________________________________________________________________________________________________________________

path_no_extension = img_path[:len(img_path)-5]      # get the path for the image without the extension
path_scanned = f'{path_no_extension}_scanned.jpg'   # add to the original image name _scanned in the end, and add the .jpg extension 
cv2.imwrite(path_scanned, warped)                   # save the scanned image to the path of the original image

print(f'Document scanned!\nSaved as {path_scanned}\n')
print("If the document wasn't properly scanned, set verbose to 1 and see if it's a problem with the original image.")
print("If not, create an issue in the github repository: https://github.com/gabrielassisdepaula/document-scanner")