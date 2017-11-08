import cv2
import argparse
import pickle

def undistort(img, mtx, dist):
    '''
    Transforms an image to compensate radial and tangential lens distortion.
    
    Input:
        img: distored image
    Output:
       undistorted_img: corrected image
    '''
    undistorted_img = cv2.undistort(img, mtx, dist, None, mtx)
    return undistorted_img

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calibrate Camera using a list of images.')
    parser.add_argument('input_image',
                        type=str,
                        help='Path to the distorted image.')
    parser.add_argument('-f', '--calibration_file',
                        type=str,
                        default='./camera_calibration.p',
                        help='Path to the camera calibration file.')
    parser.add_argument('-s', '--save',
                        action='store_true',
                        help='Whether to save the undistorted image')
    parser.add_argument('-o', '--output_image',
                        type=str,
                        default='./undistorted_image.jpg',
                        help='Saves the undistorted image to the path of your choice.')
    args = parser.parse_args()
    
    camera_calibration_file = args.calibration_file
    print('Read camera calibration data from file: ', camera_calibration_file)
    with open(camera_calibration_file, mode='rb') as f:
        cali_data = pickle.load(f)
    mtx, dist = cali_data['mtx'], cali_data['dist']
    print('Camera matrix:', mtx)
    print('Distortion coefficients: ', dist)
    
    fname = args.input_image
    print('===== Perform undistortion =====')
    print('Distorted image: ', fname)
    image = cv2.imread(fname)
    undistorted_image = undistort(image, mtx, dist)
    print('===== Done undistortion =====')
    
    output_image = args.output_image
    save_result = args.save
    if save_result == True:
        print('Save undistorted image to: ', output_image)
        cv2.imwrite(output_image, undistorted_image)