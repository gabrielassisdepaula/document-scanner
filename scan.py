from pyimagesearch.transform import four_point_transform
from skimage.filters import threshold_local
import numpy as np
import argparse, cv2, imutils

# construct the argument parser and parse the arguments (image path)
