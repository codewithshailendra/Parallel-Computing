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
    height, width = img.shape[:2]
    start_row, start_col = int(0), int(0)
    # Let's get the ending pixel coordinates (bottom right of cropped top)
    end_row, end_col = int(height * .5), int(width)
    
    img1=img[start_row:end_row , start_col:end_col]
    cv2.imshow("pp",img1)
    start_row, start_col = int(height * .5), int(0)
    # Let's get the ending pixel coordinates (bottom right of cropped bottom)
    end_row, end_col = int(height), int(width)
    
    img2=img[start_row:end_row , start_col:end_col]
    cv2.imshow("part",img2)
       
    
    """
with multiprocessing.pool.Pool() as pool:
        result1 = pool.apply_async(localcontrast, img1)
        result2 = pool.apply_async(localcontrast, img2)
        result3= pool.apply_async(localcontrast, img3)
        result4=pool.apply_async(localcontrast, img4)
        val1=result1.get()
        """
    with Executor() as executor:
        future1 = executor.submit(localcontrast, img1)
        future2 = executor.submit(localcontrast, img2)
        
        val1 = future1.result()
        
        val2=future2.result()
        
        valr=np.concatenate((val1, val2), axis=0)
        
        
        cv2.imshow("img1",val1)
        cv2.imshow("img2",val2)
        cv2.imshow("img7",valr)
        
        
    
    
    
    end=time.time()
    print(end-start)
    
