import cv2
import numpy as np
from Line import Line
from undistort import undistort
from threshold_select import rgb_to_grayscale, rgb_to_hls, hls_channel_select, mag_sobel_hls, dir_sobel_hls, sobel_hls
from perspective_transform import perspective_transform
from mask import calculate_vertices, region_of_interest

class Video():
    
    def __init__(self, cali_data):
        self.mtx = cali_data['mtx']
        self.dist = cali_data['dist']
        self.M = None
        self.Minv = None
        self.left_line = Line()
        self.right_line = Line()
        #reset the line detection after too many bad curvature detection
        self.reset = True
    
    def sliding_window(self, image):
        
        #print("sliding window")
        
        binary_warped = np.copy(image)
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
        # Create an output image to draw on and  visualize the result
        output = np.dstack((binary_warped, binary_warped, binary_warped))*255
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]/2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Choose the number of sliding windows
        nwindows = 9
        # Set height of windows
        window_height = np.int(binary_warped.shape[0]/nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []
        
        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # Draw the windows on the visualization image
            cv2.rectangle(output,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
            (0,255,0), 2) 
            cv2.rectangle(output,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
            (0,255,0), 2) 
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:        
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds] 

        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        '''
        output[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        output[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        '''
        
        self.left_line.allx = leftx
        self.left_line.ally = lefty
        self.left_line.current_fitx = left_fitx
        self.left_line.current_fit = left_fit
        self.right_line.allx = rightx
        self.right_line.ally = righty
        self.right_line.current_fitx = right_fitx
        self.right_line.current_fit = right_fit
                
    def skipping_window(self, image):
        
        #print("skipping window")
        
        binary_warped = np.copy(image)
        
        #if self.left_line.detected is True:
        #    left_fit = self.left_line.recent_fits[-1]
        #else:
        #    left_fit = self.left_line.best_fit
        #if self.right_line.detected is True:
        #    right_fit = self.right_line.recent_fits[-1]
        #else:
        #    right_fit = self.right_line.best_fit
        
        left_fit = self.left_line.best_fit
        right_fit = self.right_line.best_fit
        
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = 100
        left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
        left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
        left_fit[1]*nonzeroy + left_fit[2] + margin))) 

        right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
        right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
        right_fit[1]*nonzeroy + right_fit[2] + margin)))  

        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        '''
        # Create an image to draw on and an image to show the selection window
        output = np.dstack((binary_warped, binary_warped, binary_warped))*255
        window_img = np.zeros_like(output)
        # Color in left and right line pixels
        output[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        output[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
                                  ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
                                  ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
        output = cv2.addWeighted(output, 1, window_img, 0.3, 0)
        '''
        
        self.left_line.allx = leftx
        self.left_line.ally = lefty
        self.left_line.current_fitx = left_fitx
        self.left_line.current_fit = left_fit
        self.right_line.allx = rightx
        self.right_line.ally = righty
        self.right_line.current_fitx = right_fitx
        self.right_line.current_fit = right_fit
            
    def measuring_curvature(self):
        
        leftx = self.left_line.allx
        lefty = self.left_line.ally
        rightx = self.right_line.allx
        righty = self.right_line.ally
        
        # Define y-value where we want radius of curvature
        # Choose the maximum y-value, corresponding to the bottom of the image
        y_eval = np.max(leftx)
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/700 # meters per pixel in x dimension

        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
        right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
        # Calculate the new radii of curvature
        # Use the y-value in world space
        y_eval_m = y_eval*ym_per_pix
        left_curverad = ((1 + (2*left_fit_cr[0]*y_eval_m + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        right_curverad = ((1 + (2*right_fit_cr[0]*y_eval_m + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
        left_x_of_ymax = left_fit_cr[0]*y_eval_m**2 + left_fit_cr[1]*y_eval_m + left_fit_cr[2]
        right_x_of_ymax = right_fit_cr[0]*y_eval_m**2 + right_fit_cr[1]*y_eval_m + right_fit_cr[2]
        
        self.left_line.radius_of_curvature = left_curverad
        self.right_line.radius_of_curvature = right_curverad
        self.left_line.x_of_ymax = left_x_of_ymax        
        self.right_line.x_of_ymax = right_x_of_ymax
    
    def annotate(self, image):
        
        img = np.copy(image)
        
        # Create an image to draw the lines on
        color_warp = np.zeros_like(img).astype(np.uint8)

        # Recast the x and y points into usable format for cv2.fillPoly()        
        ploty = np.linspace(0, img.shape[0]-1, img.shape[0])
        left_fit = self.left_line.best_fit
        right_fit = self.right_line.best_fit
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, self.Minv, (img.shape[1], img.shape[0])) 
        # Combine the result with the original image
        annotated_image = cv2.addWeighted(img, 1, newwarp, 0.3, 0)

        left_curverad = self.left_line.radius_of_curvature
        right_curverad = self.right_line.radius_of_curvature
        mean_curverad = (left_curverad + right_curverad)/2
        
        left_lane_pos = self.left_line.x_of_ymax        
        right_lane_pos = self.right_line.x_of_ymax
        #Calculate the center position in meters
        center_lane_pos_m = (left_lane_pos + right_lane_pos)/2
        
        #Car center position in pixel (middle of the image)
        car_center_position = img.shape[1]/2
        #Car center position in meter
        xm_per_pix = 3.7/700 # meters per pixel in x dimension
        car_center_position_m = car_center_position*xm_per_pix
        
        center_dist = (car_center_position_m - center_lane_pos_m)
        center_dist_direction = ''
        if center_dist > 0:
            center_dist_direction = 'right'
        else:
            center_dist_direction = 'left'
        center_dis_abs = abs(center_dist)

        cv2.putText(annotated_image, text='Radius of Curvature = {0:} (m)'.format(int(mean_curverad)), 
                    org=(100,100), fontFace=2, fontScale=1.5,
                    color=(255,255,255), thickness=2)
        cv2.putText(annotated_image, 
                    text='Vehicle Position = {0:.2f} (m) {1} of the center'.format(center_dis_abs, center_dist_direction),
                    org=(100,150), fontFace=2, fontScale=1.5,
                    color=(255,255,255), thickness=2)
        return annotated_image
        
    def process(self, image):
        
        # Undistort image
        undistorted_image = undistort(image, self.mtx, self.dist)
        
        # Threshold select
        hls = rgb_to_hls(undistorted_image)
        hls_binary = hls_channel_select(hls, thresh=(130, 255), channel=2)
        x_sobel_binary = sobel_hls(hls, thresh=(35,150), gradient='x', channel=1)
        binary_select = cv2.bitwise_or(hls_binary, x_sobel_binary)
        
        # Mask
        vertices = calculate_vertices(binary_select)
        masked_edges = region_of_interest(binary_select, vertices)
        
        # Perspective Transform
        binary_warped, self.M, self.Minv = perspective_transform(masked_edges)
        
        # If reset, redo the sliding window search
        # Else just search in a margin around the previous line position
        if self.reset is True:
            self.sliding_window(binary_warped)
        else:
            self.skipping_window(binary_warped)
        
        self.measuring_curvature()
        self.left_line.update(self.reset)
        self.right_line.update(self.reset)
        
        max_bad_detection = 8
        if self.left_line.bad_detection > max_bad_detection or self.right_line.bad_detection > max_bad_detection:
            self.reset = True
            #print('Debug: Set rest to True')
        else:
            self.reset = False
            #print('Debug: Set reset to False')
        
        processed_image = self.annotate(undistorted_image)
        
        return processed_image