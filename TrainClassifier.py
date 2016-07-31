# this program trains our classifier for handwritten dataset and outputs characters_value.pkl
# which will be used by PerformRecognition.py for predicting character

import sys,cv2,operator,os
import numpy as np
from sklearn.externals import joblib
from sklearn.svm import LinearSVC

######################################### GLOBAL variables ####################################
MIN_CONTOUR_AREA = 100

DEFAULT_IMG_WID = 20
DEFAULT_IMG_HEIGHT = 30

##################################Training Function ###########################################
def train():
    path="/home/absolutezero/Downloads/project_charachter_recognition_nochange/dataset"

    """ ML approach - Using Linear SVM for training """

    clf = LinearSVC()

    """ values for each character found in given image """

    values=[]

    """ all characters found in given image """

    allCharacters =  np.empty((0, DEFAULT_IMG_WID * DEFAULT_IMG_HEIGHT))

    """ training every character one by one  """

    for x in xrange(65,91):
        charactersImage = cv2.imread(os.path.join(path,chr(x)+'.png'))  # read in characters image

        if charactersImage is None:                             # if image was not read successfully
            print "error: image not read from file \n\n"        # print error message to std out
            os.system("pause")                                  # pause so user can see error message
            return                                              # and exit function (which exits program)

        grayImage = cv2.cvtColor(charactersImage, cv2.COLOR_BGR2GRAY)               # get grayscale image
        blurredImage = cv2.GaussianBlur(grayImage, (5,5), 0)                        # blur the image

                                                                                # filter image from grayscale to black and white
        thresholdImage = cv2.adaptiveThreshold(blurredImage,                    # input image
                                          255,                                  # make pixels that pass the threshold full white
                                          cv2.ADAPTIVE_THRESH_GAUSSIAN_C,       # use gaussian rather than mean, seems to give better results
                                          cv2.THRESH_BINARY_INV,                # invert so foreground will be white, background will be black
                                          11,                                   # size of a pixel neighborhood used to calculate threshold value
                                          2)                                    # constant subtracted from the mean or weighted mean

        cv2.imshow("thresholdImage", thresholdImage)      # show threshold image for reference

        thresholdImageCopy = thresholdImage.copy()        # make a copy of the thresh image, this in necessary b/c findContours modifies the image

        imgContours, imgHierarchy = cv2.findContours(thresholdImageCopy,        # input image, make sure to use a copy since the function will modify this image in the course of finding contours
                                                     cv2.RETR_EXTERNAL,         # retrieve the outermost contours only
                                                     cv2.CHAIN_APPROX_SIMPLE)   # compress horizontal, vertical, and diagonal segments and leave only their end points

        for imgContour in imgContours:                                          # for each contour
            if cv2.contourArea(imgContour) > MIN_CONTOUR_AREA:                  # if contour is big enough to consider
                [X, Y, W, H] = cv2.boundingRect(imgContour)                     # get and break out bounding rect

                                                            # draw rectangle around each contour as we ask user for input
                cv2.rectangle(charactersImage,              # draw rectangle on original training image
                              (X, Y),                       # upper left corner
                              (X+W,Y+H),                    # lower right corner
                              (0, 0, 255),                  # red
                              2)                            # thickness

                characterImg = thresholdImage[Y:Y+H, X:X+W]                             # crop char out of threshold image
                characterImgResized = cv2.resize(characterImg, (DEFAULT_IMG_WID, DEFAULT_IMG_HEIGHT))     # resize image, this will be more consistent for recognition and storage

                # cv2.imshow("characterImg", characterImg)                    # show cropped out char for reference
                # cv2.imshow("characterImgResized", characterImgResized)      # show resized image for reference
                # cv2.imshow("training_numbers.png", charactersImage)      # show training numbers image, this will now have red rectangles drawn on it

                CharacterValue = x

                # if intChar == 27:                   # if esc key was pressed
                #     sys.exit()                      # exit program
                # elif intChar in intValidChars:      # else if the char is in the list of chars we are looking for . . .

                values.append(CharacterValue)                                                # append classification char to integer list of chars (we will convert to float later before writing to file)

                finalCharacterImage = characterImgResized.reshape((1, DEFAULT_IMG_WID * DEFAULT_IMG_HEIGHT))  # flatten image to 1d numpy array so we can write to file later
                allCharacters = np.append(allCharacters, finalCharacterImage, 0)                    # add current flattened impage numpy array to list of flattened image numpy arrays

    floatValues = np.array(values, np.float32)                   # convert values list of ints to numpy array of floats

    finalValues = floatValues.reshape((floatValues.size, 1))   # flatten numpy array of floats to 1d so we can write to file later

    print "\n\ntraining complete !!\n"

    np.savetxt("classifications.txt", finalValues)
    np.savetxt("flattened_images.txt", allCharacters)

    # clf = LinearSVC()
    # labels = finalValues
    # features = allCharacters
    # Perform the training
    # print labels
    # print features
    clf.fit(allCharacters, finalValues.ravel())

    joblib.dump((clf), "characters_value.pkl", compress=3)

    cv2.destroyAllWindows()
    return None

###################################################################################################
if __name__ == "__main__":
    train()
