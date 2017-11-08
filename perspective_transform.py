import numpy as np
import cv2

def perspective_transform(image):
    '''
    Perform perspective transform
    Input:
      image: image before perspective transform
    Output:
      warped_image: image after perspective transform
    '''
    src = np.float32([[200, 720],[1100, 720],[550, 480],[735, 480]])
    dst = np.float32([[300, 720],[1000, 720],[300, 0],[1000, 0]])
    
    M = cv2.getPerspectiveTransform(src, dst)
    img_size = (image.shape[1], image.shape[0])
    warped_image = cv2.warpPerspective(image, M, img_size, flags=cv2.INTER_LINEAR)
    return warped_image