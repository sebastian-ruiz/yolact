# Calculate oriented bounding boxes for sets of points
# and for binary mask images/label images
# Volker.Hilsenstein@monash.edu
# This is based on the following stackoverflow answer
# by stackoverflow user Quasiomondo (Mario Klingemann)
# https://stackoverflow.com/questions/32892932/create-the-oriented-bounding-box-obb-with-python-and-numpy

import numpy as np
import skimage.morphology
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
import cv2
import os
import time


def get_obb_using_cv(contour, img=None):

    if contour is None or len(contour) < 4:
        return None, None, None
    
    # https://stackoverflow.com/questions/18207181/opencv-python-draw-minarearect-rotatedrect-not-implemented
    # (center(x, y), (width, height), angle of rotation) = cv2.minAreaRect(points)
    rect = cv2.minAreaRect(contour) 
    box = np.int0(cv2.boxPoints(rect))
    center = np.int0(rect[0])
    changing_width, changing_height = box_w_h(box)
    changing_rot = rect[2]
    
    #  rotation is a bit funny, and we correct for it. see here:
    # https://stackoverflow.com/questions/15956124/minarearect-angles-unsure-about-the-angle-returned
    if changing_height > changing_width:
        correct_rot = changing_rot
    else:
        correct_rot = changing_rot - 90

    correct_height = changing_height
    correct_width = changing_width
    if changing_height < changing_width:
        correct_height = changing_width
        correct_width = changing_height
    
    # if the width or height of the rectangle is 0, then we return None
    if np.isclose(correct_height, 0.0) or np.isclose(correct_width, 0.0):
        return None, None, None

    # clip so that the box is still in the bounds of the img
    if img is not None:
        # invert img.shape because points are x, y
        box = clip_box_to_img_shape(box, img.shape)
        # box = np.clip(box, a_min=np.asarray([0, 0]), a_max=np.asarray(img.shape[:2])[::-1] - 1) 
    
    return box, center, correct_rot

def box_w_h(box):
    return np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[1] - box[2])

def clip_box_to_img_shape(box, img_shape):
    box = np.clip(box, a_min=np.asarray([0, 0]), a_max=np.asarray(img_shape[:2])[::-1] - 1) 
    return box


def get_obb_from_contour(contour, img=None):
    """ given a binary mask, calculate the oriented 
    bounding box of the foreground object. This is done
    by calculating the outline of the object, transform
    the pixel coordinates of the outline into a list of
    points and then calling :func: `get_obb_from_points`

    Parameters:
        mask_im: binary numpy array

    """
    
    # https://stackoverflow.com/questions/13542855/algorithm-to-find-the-minimum-area-rectangle-for-given-points-in-order-to-comput/33619018#33619018
    corners, center, rot = get_obb_using_cv(contour, img)
    # corners, center, rot_quat =  get_obb_using_eig(contour)

    # better_rot_quat = rot_quat
    # if corners is not None:
    #     better_rot_quat = better_quaternion(corners)
    return corners, center, rot

def get_obb_from_mask(mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) > 0:
        # get the contour with the largest area. Assume this is the one containing our object
        cnt = max(cnts, key = cv2.contourArea)
        mask_contour = np.squeeze(cnt)
        
        return get_obb_from_contour(mask_contour)

    return None, None, None