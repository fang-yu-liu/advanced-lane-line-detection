import argparse
import glob
import numpy as np
import cv2
import pickle

def findObjpointsAndImgpoints(images):
    '''
    Find object points and image points from a list of images for perspective transform
    
    Input:
        images: list of images
    Output:
        objpoints: 3d points in real world space (chessboard corners)
        imgpoints: 2d points in image plane (chessboard corners)
    '''
    nx = 9
    ny = 6
    objp = np.zeros((ny*nx,3), np.float32)
    objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    # Step through the list and search for chessboard corners
    for idx, img in enumerate(images):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
    return objpoints, imgpoints

def calibrate_camera(images):
    '''
    Calculate camera matrix and distortion coefficients using a list of images
    
    Input:
        images: list of images
    Output:
        mtx: camera matrix
        dist: distortion coefficients
    '''
    objpoints, imgpoints = findObjpointsAndImgpoints(images)
    
    img = images[0]
    img_size = (img.shape[1], img.shape[0])
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
    return mtx, dist

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calibrate Camera using a list of images.')
    parser.add_argument('-i', '--input_folder',
                        type=str,
                        default='./camera_cal/',
                        help='Path to image folder. The calibration will be based upon those images.')
    parser.add_argument('-s', '--save',
                        action='store_true',
                        help='Whether to save the calibration result or not')
    parser.add_argument('-o', '--output_file',
                        type=str,
                        default='camera_calibration.p',
                        help='Saves the output to a name of your choice')
    args = parser.parse_args()
    
    input_folder = args.input_folder
    output_file = args.output_file
    save_result = args.save
    
    print('Read images from: ', input_folder)
    images_filename = glob.glob(input_folder + 'calibration*.jpg')
    if not images_filename:
        print('No images found in the folder: ', input_folder)
        exit()
        
    print('Images for calibration: ', images_filename)
    images = [cv2.imread(fname) for fname in images_filename]
    print('===== Start calibrating =====')
    mtx, dist = calibrate_camera(images)
    print('Camera matrix:', mtx)
    print('Distortion coefficients: ', dist)
    print('===== Done calibrating =====')
    
    if save_result == True:
        print('Save calibration result to: ', output_file)
        # Save the camera calibration result for later use
        cali_data = {}
        cali_data["mtx"] = mtx
        cali_data["dist"] = dist
        with open(output_file, mode='wb') as f:
            pickle.dump(cali_data, f)