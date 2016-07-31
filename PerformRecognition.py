# TrainAndTest.py
import cv2
import numpy as np
import operator
import os
from sklearn.externals import joblib


# GLOBAL variables ##########################################################################

MIN_CONTOUR_AREA = 100

DEFAULT_IMG_WID = 20
DEFAULT_IMG_HEIGHT = 30

################################# An Object for storing contour information ###################################
class ContourWithData():

    def __init__(self,imgContour):
        self.imageContour = imgContour
        self.boundingRect = cv2.boundingRect(self.imageContour)
        [X, Y, Width, Height] = self.boundingRect
        self.rectX = X
        self.rectY = Y
        self.rectWidth = Width
        self.rectHeight = Height
        self.contourAREA = cv2.contourArea(self.imageContour)

    def checkIfContourIsValid(self):
        if self.contourAREA < MIN_CONTOUR_AREA: return False
        return True

############################### Function to predict characters in image#################################
def predict():
    allContoursWithData = []                                          # declare empty lists,
    validContoursWithData = []                                        # we will fill these shortly
    clf = joblib.load("characters_value.pkl")



    imgTestingCharacters = cv2.imread("hogya.png")                    # read in testing numbers image

    if imgTestingCharacters is None:                                  # if image was not read successfully
        print "error: image not read from file \n\n"                  # print error message to std out
        os.system("pause")                                            # pause so user can see error message
        return                                                        # and exit function (which exits program)


    grayImage = cv2.cvtColor(imgTestingCharacters, cv2.COLOR_BGR2GRAY)      # get grayscale image
    blurredImage = cv2.GaussianBlur(grayImage, (5,5), 0)                    # blur the image

                                                                            # filter image from grayscale to black and white
    thresholdImage = cv2.adaptiveThreshold(blurredImage,                    # input image
                                      255,                                  # make pixels that pass the threshold full white
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C,       # use gaussian rather than mean, seems to give better results
                                      cv2.THRESH_BINARY_INV,                # invert so foreground will be white, background will be black
                                      11,                                   # size of a pixel neighborhood used to calculate threshold value
                                      2)                                    # constant subtracted from the mean or weighted mean

    thresholdImageCopy = thresholdImage.copy()        # make a copy of the thresh image, this in necessary b/c findContours modifies the image

    imgContours, imgHierarchy = cv2.findContours(thresholdImageCopy,        # input image, make sure to use a copy since the function will modify this image in the course of finding contours
                                                 cv2.RETR_EXTERNAL,         # retrieve the outermost contours only
                                                 cv2.CHAIN_APPROX_SIMPLE)   # compress horizontal, vertical, and diagonal segments and leave only their end points

    for imgContour in imgContours:                                          # for each contour
        contourWithData = ContourWithData(imgContour)                       # instantiate a contour with data object
        allContoursWithData.append(contourWithData)                         # add contour with data object to list of all contours with data

    for contourWithData in allContoursWithData:                             # for all contours
        if contourWithData.checkIfContourIsValid():                         # check if valid
            validContoursWithData.append(contourWithData)                   # if so, append to valid contour list

    validContoursWithData.sort(key = operator.attrgetter("rectX"))          # sort contours from left to right

    strFinalString = ""         # declare final string, this will have the final number sequence by the end of the program

    for contour in validContoursWithData:                                   # for each contour
                                                                            # draw a green rect around the current char
        cv2.rectangle(imgTestingCharacters,                                 # draw rectangle on original testing image
                      (contour.rectX, contour.rectY),                       # upper left corner
                      (contour.rectX + contour.rectWidth, contour.rectY + contour.rectHeight),        # lower right corner
                      (0, 255, 0),                                          # green
                      2)                                                    # thickness

        characterImg = thresholdImage[contour.rectY : contour.rectY + contour.rectHeight,             # crop char out of threshold image
                           contour.rectX : contour.rectX + contour.rectWidth]

        characterImgResized = cv2.resize(characterImg, (DEFAULT_IMG_WID, DEFAULT_IMG_HEIGHT))         # resize image, this will be more consistent for recognition and storage

        finalCharacterImage = characterImgResized.reshape((1, DEFAULT_IMG_WID * DEFAULT_IMG_HEIGHT))  # flatten image into 1d numpy array

        finalCharacterImage = np.float32(finalCharacterImage)             # convert from 1d numpy array of ints to 1d numpy array of floats
        temp = clf.predict(finalCharacterImage)

        charValue=str(unichr(int(temp)))                                  # append current char to full string
        strFinalString = strFinalString + charValue

    print "\n" + strFinalString + "\n"                                    # show the full string

    cv2.imshow("imgTestingCharacters", imgTestingCharacters)              # show input image with green boxes drawn around found digits
    cv2.waitKey(0)                                                        # wait for user key press

    cv2.destroyAllWindows()                                               # remove windows from memory

    return None

###################################################################################################
if __name__ == "__main__":
    predict()
