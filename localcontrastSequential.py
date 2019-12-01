import cv2
import numpy as np
import time
from multiprocessing.pool import Pool
from multiprocessing import Process
import multiprocessing
from concurrent.futures import ProcessPoolExecutor as Executor
start=time.time()
img = cv2.imread('finalimg.jpg')
def localcontrast(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    float_gray = gray.astype(np.float32) / 255.0
    blur = cv2.GaussianBlur(float_gray, (0, 0), sigmaX=2, sigmaY=2)
    num = float_gray - blur
    blur = cv2.GaussianBlur(num*num, (0, 0), sigmaX=20, sigmaY=20)
    den = cv2.pow(blur, 0.5)
    gray = num / den
    cv2.normalize(gray, dst=gray, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX)
    cv2.imwrite("localcontrast.jpg", gray * 255)
    image=cv2.imread("localcontrast.jpg")
    return image
if __name__=="__main__":
    img = cv2.imread('finalimg.jpg')
    final=localcontrast(img)
    cv2.imshow("final",final)
        
    
    
    
    end=time.time()
    print(end-start)
    
