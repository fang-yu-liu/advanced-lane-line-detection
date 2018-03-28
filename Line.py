import numpy as np
import cv2

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        #n iteration that we want to average the line information
        self.n = 5
        # was the line detected in the last iteration?
        self.detected = False
        # numbers of the bad detection in a row
        self.bad_detection = 0
        # max allowable bad detection in a row
        self.max_bad_detection = 5
        #polynomial coefficients of the last n iterations
        self.recent_fits = []
        #x values of the last n fits of the line
        self.recent_fitsx = []
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        #average x values of the fitted line over the last n iterations
        self.best_fitx =None
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        #x values of the last n fits of the most recent fit
        self.current_fitx = [np.array([False])]
        #radius of curvature of the line in some units
        self.radius_of_curvature = None
        #the x value with max y-value
        self.x_of_ymax = None
        #distance in meters of vehicle center from the line
        self.line_base_pos = None
        #difference in fit coefficients between best and current fits
        self.diffs = np.array([0,0,0], dtype='float')
        #threshod for the difference in fit coefficients
        self.thresh = np.array([2,2,2], dtype='float')
        #x values for detected line pixels
        self.allx = None
        #y values for detected line pixels
        self.ally = None
        
    def add_to_recent_fits(self, fit, fitx):
        self.recent_fitsx.append(fitx)
        if len(self.recent_fitsx) > self.n:
            # only keep newest n iterations, remove the oldest one (index = 0)
            self.recent_fitsx.pop(0)
        self.best_fitx = np.mean(self.recent_fitsx,0)
        
        self.recent_fits.append(fit)
        if len(self.recent_fits) > self.n:
            # only keep newest n iterations, remove the oldest one (index = 0)
            self.recent_fits.pop(0)
        self.best_fit = np.mean(self.recent_fits,0)
        
    def update(self, reset):
        if reset is True:
            self.recent_fits = []
            self.bad_detection = 0
            #print("Reset the recent fits")
            
        if self.current_fit is None:
            self.detected = False
            #print("No current fit is found")
        else:
            self.detected = True
            if self.best_fit is not None:
                self.diffs = abs(self.best_fit - self.current_fit)
                if np.all(self.diffs < self.thresh):
                    #print("Good new measurement, add to recent fits")
                    self.bad_detection = 0
                    self.add_to_recent_fits(self.current_fit, self.current_fitx)
                else:
                    #print(self.bad_detection)
                    self.bad_detection += 1
                    #print("Curvature changes a lot from the previous measurement")
                    if self.bad_detection > self.max_bad_detection:
                        #print("set detected to false")
                        self.detected = False
                
            else: 
                self.add_to_recent_fits(self.current_fit, self.current_fitx)
                #print("No best fit yet, add the current one to recent fits")
   
