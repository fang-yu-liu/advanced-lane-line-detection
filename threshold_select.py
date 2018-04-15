import cv2
import numpy as np

def rgb_to_grayscale(img):
    '''
    Transform an image from BGR to GRAYSCALE
    Input:
        img: image in BGR colorspace
    Output:
        gray: image in grayscale colorspace
    '''
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(float)
    return gray

def rgb_to_hls(img):
    '''
    Transform an image from BGR to GRAYSCALE
    Input:
        img: image in BGR colorspace
    Output:
        hls: image in HLS colorspace
    '''
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(float)
    return hls

def hls_channel_select(hls, thresh=(0, 255), channel=2):
    '''
    Thresholds the S-channel of HLS
    
    Input:
        hls: image in HLS colorspace
        thresh: lower and upper bound of the threshold
        channel: the channel to perform threshold select 0-H, 1-L, 2-S (default=2)
    Output:
        hls_binary: binary output of the threshold select
    '''
    channel_select = hls[:,:,channel]
    hls_binary = np.zeros_like(channel_select)
    hls_binary[(channel_select > thresh[0]) & (channel_select <= thresh[1])] = 1
    return hls_binary

def mag_sobel_hls(hls, thresh=(0, 255), channel=2, sobel_kernel=3):
    '''
    Thresholds the magnitude of the gradient for a given sobel kernel size
    
    Input:
        hls: image in HLS colorspace
        thresh: lower and upper bound of the threshold (default=(0,255))
        channel: the channel to perform threshold select 0-H, 1-L, 2-S (default=2)
        sobel_kernel: sobel kernel size (default=3)
    Output:
        binary_output: binary output of the threshold select
    '''
    channel_select = hls[:,:,channel]
    sobelx = cv2.Sobel(channel_select, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(channel_select, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    mag_sobel = np.absolute(np.sqrt(sobelx**2, sobely**2))
    scaled_sobel = np.uint8(255*mag_sobel/np.max(mag_sobel))
    
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel > thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return binary_output

def dir_sobel_hls(hls, thresh=(0, np.pi/2), channel=2, sobel_kernel=3):
    '''
    Thresholds the direction of the gradient for a given sobel kernel size
    
    Input:
        hls: image in HLS colorspace
        thresh: lower and upper bound of the threshold (default=(0,255))
        channel: the channel to perform threshold select 0-H, 1-L, 2-S (default=2)
        sobel_kernel: sobel kernel size (default=3)
    Output:
        binary_output: binary output of the threshold select
    '''
    channel_select = hls[:,:,channel]
    sobelx = cv2.Sobel(channel_select, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(channel_select, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    dir_sobel = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    
    binary_output = np.zeros_like(dir_sobel)
    binary_output[(dir_sobel > thresh[0]) & (dir_sobel <= thresh[1])] = 1
    return binary_output
    
def sobel_hls(hls, thresh=(0, 255), gradient='x', channel=2, sobel_kernel=3):
    '''
    Thresholds the gradient for a given sobel kernel size
    
    Input:
        hls: image in HLS colorspace
        thresh: lower and upper bound of the threshold (default=(0,255))
        gradient: the gradient direction 'x' or 'y' (default='x')
        channel: the channel to perform threshold select 0-H, 1-L, 2-S (default=2)
        sobel_kernel: sobel kernel size (default=3)
    Output:
        binary_output: binary output of the threshold select
    '''
    channel_select = hls[:,:,channel]
    if gradient == 'x':
        sobel = cv2.Sobel(channel_select, cv2.CV_64F, 1, 0, ksize=sobel_kernel) # Take the derivative in x
    if gradient == 'y':
        sobel = cv2.Sobel(channel_select, cv2.CV_64F, 0, 1, ksize=sobel_kernel) # Take the derivative in y
    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    
    binary_output = np.zeros_like(channel_select)
    binary_output[(scaled_sobel > thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return binary_output