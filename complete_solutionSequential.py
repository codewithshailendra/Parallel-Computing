# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 10:56:28 2019

@author: dell
"""# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 19:26:55 2019

@author: dell
"""

from PIL import Image
from skimage import io, color
import cv2 as cv
import cv2
import numpy as np
import sys
import argparse
import math
from multiprocessing.pool import Pool
from multiprocessing import Process

def applyCLAHE(img,l):
    l1,a1,b1=cv2.split(img)
    clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(3,3))
    l2 = clahe.apply(l)
    lab2=cv2.merge((l2,a1,b1))
    img2 = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
    return (img2,l2)


def localcontrast(gray):
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    float_gray = gray.astype(np.float32) / 255.0
    blur = cv2.GaussianBlur(float_gray, (0, 0), sigmaX=2, sigmaY=2)
    num = float_gray - blur
    blur = cv2.GaussianBlur(num*num, (0, 0), sigmaX=20, sigmaY=20)
    den = cv2.pow(blur, 0.5)
    gray = cv2.divide(num,den)
    cv2.normalize(gray, dst=gray, alpha=0.0,
                  beta=1.0, norm_type=cv2.NORM_MINMAX)
    #cv2.imwrite("white1w2.jpg", gray * 255)

    return gray


def saliency(image):
    # initialize OpenCV's static saliency spectral residual detector and
    # compute the saliency map
    saliency = cv.saliency.StaticSaliencySpectralResidual_create()
    (success, saliencyMap) = saliency.computeSaliency(image)
    return saliencyMap


    

def laplacian_function(img):
    # [variables]
    # Declare the variables we are going to use
    ddepth = cv.CV_16S

    kernel_size = 3
    window_name = "Laplace Demo"
    # [variables]
    # [load]
    
    src = img

    # Check if image is loaded fine
    if src is None:
        print('Error opening image')
        print('Program Arguments: [image_name -- default ../data/lena.jpg]')
        return -1
    # [load]
    # [reduce_noise]
    # Remove noise by blurring with a Gaussian filter
    
    for i in range(1, kernel_size, 2):
        src = cv.bilateralFilter(src, i, i * 2, i / 2)

    # [reduce_noise]
    # [convert_to_gray]
    # Convert the image to grayscale
    src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    # [convert_to_gray]
    # Create Window
    #cv.namedWindow(window_name, cv.WINDOW_AUTOSIZE)
    # [laplacian]
    # Apply Laplace function
    dst = cv.Laplacian(src_gray, ddepth,  kernel_size)
    # [laplacian]
    # [convert]
    # converting back to uint8
    abs_dst = cv.convertScaleAbs(dst)
    return abs_dst

def Exposedness(image):
    sigma = 0.25
    average = 0.5

    image=image.astype(float)
    image=image/255.0
    rows = image.shape[0]
    cols = image.shape[1]

    exposedness = np.zeros((rows, cols))
    exposedness = image

    for r in range(rows):
        for c in range(cols):
            value = np.exp(
                (-1)*(np.power(image[r][c]-average, 2.0))/(2*np.power(sigma, 2.0)))
            #print(type(value))
            exposedness[r][c] = value
    #exposedness=exposedness*255
    #exposedness.astype(np.uint8)
    return exposedness

"""
def Exposedness(image):
    sigma = 0.25
    average = 0.5

    rows = image.shape[0]
    cols = image.shape[1]
    no_of_channels = image.shape[2]

    exposedness = np.zeros((rows, cols, no_of_channels))
    exposedness = image

    for r in range(rows):
        for c in range(cols):
            value = np.exp(
                (-1)*(np.power(image[r][c][0]-average, 2.0))/(2*np.power(sigma, 2.0)))
            #print(type(value))
            exposedness[r][c][0] = value
 
    return exposedness

"""
def laplacian(img):
    laplacian = cv.Laplacian(img,-1)
    return laplacian




def color_balance(img, percent):
    if percent <= 0:
        percent = 5  # taken as an average of (1-10).

    rows = img.shape[0]
    cols = img.shape[1]
    # knowing the no. of channels in the present image
    no_of_chanl = img.shape[2]

    # halving the given percentage based on the given research paper
    halfpercent = percent/200.0

    # list for storing all the present channels of the image separately.
    channels = []

    if no_of_chanl == 3:
        for i in range(3):
            # add all the present channels of the image to this list separately
            channels.append(img[:, :, i:i+1])
    else:
        channels.append(img)

    results = []

    for i in range(no_of_chanl):
        #print(channels[i].shape)
        plane = channels[i].reshape(1, rows*cols, 1)
        plane.sort()
        lower_value = plane[0][int(plane.shape[1]*halfpercent)][0]
        top_value = plane[0][int(plane.shape[1]*(1-halfpercent))][0]

        channel = channels[i]

        for p in range(rows):
            for q in range(cols):
                if channel[p][q][0] < lower_value:
                    channel[p][q][0] = lower_value
                if channel[p][q][0] < top_value:
                    channel[p][q][0] = top_value

        channel = cv2.normalize(channel, None, 0.0, 255.0/2, cv2.NORM_MINMAX)
        # convert the image in desired format-converted

        results.append(channel)

    output_image = np.zeros((rows, cols, 3))
    
    output_image = cv2.merge(results)
    return output_image


MAX_KERNEL_LENGTH = 40 
src = None
dst = None


def coherent_noisereduction(img):
    # Load the source image
    global src
    src = img
    global dst
    dst = np.copy(src)
    for i in range(1, MAX_KERNEL_LENGTH, 2):
        dst = cv.bilateralFilter(src, i, i * 2, i / 2)
        #cv.imwrite('result.jpg', dst)
    return dst

imgpath1="2.jpg"
#input_image = cv2.imread(
#    "F:\\cs\\Data Science\\computer vision\\Underwater_images\\3 wall divers.jpg",1)
input_image = cv2.imread(imgpath1)
input_image = cv2.resize(input_image,(512,512),interpolation=cv2.INTER_AREA)

input_image_1 = color_balance(input_image,100)
input_image_1=input_image_1.astype(np.uint8)
lab1 = cv2.cvtColor(input_image_1, cv2.COLOR_BGR2LAB)
l1,a1,b1=cv2.split(lab1)
input_image_2,l2=applyCLAHE(lab1,l1)


lumnc1 = cv2.cvtColor(input_image_1, cv2.COLOR_BGR2LAB)
lumnc2 = cv2.cvtColor(input_image_2, cv2.COLOR_BGR2LAB)

enhance_1=input_image_1



def GaussianPyramid(img, level):
    g = img.copy()
    gp = [g]
    for i in range(level):
        g = cv2.pyrDown(g)
        gp.append(g)
    return gp





def LaplacianPyramid(img, level):
    l = img.copy()
    gp = GaussianPyramid(img, level)
    lp = [gp[level]]
    for i in range(level, 0, -1):
        size = (gp[i-1].shape[1], gp[i-1].shape[0])
        ge = cv2.pyrUp(gp[i], dstsize=size)
        l = cv2.subtract(gp[i-1], ge)
        lp.append(l)
    lp.reverse()
    return lp


def PyramidReconstruct(lapl_pyr):
  output = None
  output = np.zeros(
      (lapl_pyr[0].shape[0], lapl_pyr[0].shape[1]), dtype=np.float64)
  for i in range(len(lapl_pyr)-1, 0, -1):
    lap = cv2.pyrUp(lapl_pyr[i])
    lapb = lapl_pyr[i-1]
    if lap.shape[0] > lapb.shape[0]:
      lap = np.delete(lap, (-1), axis=0)
    if lap.shape[1] > lapb.shape[1]:
      lap = np.delete(lap, (-1), axis=1)
    tmp = lap + lapb
    lapl_pyr.pop()
    lapl_pyr.pop()
    lapl_pyr.append(tmp)
    output = tmp   
  return output



def Fusion(w1, w2, img1, img2):
    level = 5
    weight1 = GaussianPyramid(w1, level)
    weight2 = GaussianPyramid(w2, level)
    img1=img1.astype(float)
    img2=img2.astype(float)
    b1, g1, r1 = cv2.split(img1)
    b_pyr1 = LaplacianPyramid(b1, level)
    g_pyr1 = LaplacianPyramid(g1, level)
    r_pyr1 = LaplacianPyramid(r1, level)
    b2, g2, r2 = cv2.split(img2)
    b_pyr2 = LaplacianPyramid(b2, level)
    g_pyr2 = LaplacianPyramid(g2, level)
    r_pyr2 = LaplacianPyramid(r2, level)
    b_pyr = []
    g_pyr = []
    r_pyr = []
    for i in range(level):
        b_pyr.append(cv2.add(cv2.multiply(
            weight1[i], b_pyr1[i]), cv2.multiply(weight2[i], b_pyr2[i])))
        g_pyr.append(cv2.add(cv2.multiply(
            weight1[i], g_pyr1[i]), cv2.multiply(weight2[i], g_pyr2[i])))
        r_pyr.append(cv2.add(cv2.multiply(
            weight1[i], r_pyr1[i]), cv2.multiply(weight2[i], r_pyr2[i])))
    b_channel = PyramidReconstruct(b_pyr)
    g_channel = PyramidReconstruct(g_pyr)
    r_channel = PyramidReconstruct(r_pyr)
    out_img = cv2.merge((b_channel, g_channel, r_channel))
    out_img=out_img.astype(np.uint8)
    return out_img

l1=l1/255.0
l1=l1.astype(float)
laplcn_w1 = laplacian(l1).astype(float)
loclcontrst_w1 = localcontrast(l1).astype(float)
#slncy_w1 = saliency(input_image_1).astype(float)
expsd_w1 = Exposedness(l1).astype(float)


l2=l2/255.0
l2=l2.astype(float)
laplcn_w2 = laplacian(l2).astype(float)
loclcontrst_w2 = localcontrast(l2).astype(float)
#slncy_w2 = saliency(input_image_2).astype(float)
expsd_w2 = Exposedness(l2).astype(float) 






def calcweight1(l):
    
    
    weight = laplcn_w1.copy()
    weight = cv2.add(weight, loclcontrst_w1)
    #weight = cv2.add(weight, slncy_w1)
    weight = cv2.add(weight, expsd_w1)
    return weight


def calcweight2(l):
    
    
    weight = laplcn_w2.copy()
    weight = cv2.add(weight, loclcontrst_w2)
    ##weight = cv2.add(weight, slncy_w2)
    weight = cv2.add(weight, expsd_w2)
    return weight


# luminance
    """
lumnc1 = cv2.cvtColor(input_image_1, cv2.COLOR_BGR2GRAY)
lumnc2 = cv2.cvtColor(input_image_2, cv2.COLOR_BGR2GRAY)
"""
#lab1 = color.rgb2lab(input_image_1)


"""
lumnc1=rgb2lab(input_image_1)
lumnc2=rgb2lab(input_image_2)
"""

def enhance(img,level):
    global w1
    global w2
    w1=calcweight1(l1)
    w2=calcweight2(l2)
    print(w1)
    print(w2)
   
    w_sum=cv2.add(w1,w2)
    w1=cv2.divide(w1,w_sum)
    w2=cv2.divide(w2,w_sum)
    print(w1)
    print(w2)
    
    return Fusion(w1,w2,input_image_1,input_image_2)










#weight_1 = calcweight1(lumnc)
#weight_2 = calcweight2(lumnc)    


#Gaussianp_1=GaussianPyramid(weight_1,5)
#Gaussianp_2=GaussianPyramid(weight_2,5)

#laplcn_1=LaplacianPyramid(input_image_1)
#laplcn_2=LaplacianPyramid(input_image_2)
    



final_image=enhance(input_image,5)
cv2.imshow("original_image",input_image)
cv2.waitKey(0)
cv2.imshow("enhance_image_1",enhance_1)
cv2.waitKey(0)
cv2.imshow("enhance_2",final_image)
cv2.waitKey(0)




cv2.destroyAllWindows()


"""
cv2.imshow("original_image",l1 )
cv2.waitKey(0)
cv2.imshow("fianl_image",saliency(input_image_1))
cv2.waitKey(0)
cv2.destroyAllWindows()


def main():
    w1=calcweight1(lumnc1)
    w2=calcweight2(lumnc2)
    w_sum=cv2.add(w1,w2)
    w1=cv2.divide(w1,w_sum)
    w2=cv2.divide(w2,w_sum)
    print(w2[100][80])
    cv2.imshow("original_image",w2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
if __name__=="__main__":
    main()
"""


