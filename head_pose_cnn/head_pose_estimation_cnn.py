'''
Lam Huynh, Oulu 2017
Head pose estimation using trained CNN model, with haar for face detection
'''
print("New Oulu")

import tensorflow as tf
import numpy as np
import cv2
import sys
from deepgaze.haar_cascade import haarCascade
from deepgaze.face_landmark_detection import faceLandmarkDetection


#If True enables the verbose mode
DEBUG = False

#Antropometric constant values of the human head.
#Found on wikipedia and on:
# "Head-and-Face Anthropometric Survey of U.S. Respirator Users"
#
#X-Y-Z with X pointing forward and Y on the left.
#The X-Y-Z coordinates used are like the standard
# coordinates of ROS (robotic operative system)
P3D_RIGHT_SIDE = np.float32([-100.0, -77.5, -5.0]) #0
P3D_GONION_RIGHT = np.float32([-110.0, -77.5, -85.0]) #4
P3D_MENTON = np.float32([0.0, 0.0, -122.7]) #8
P3D_GONION_LEFT = np.float32([-110.0, 77.5, -85.0]) #12
P3D_LEFT_SIDE = np.float32([-100.0, 77.5, -5.0]) #16
P3D_FRONTAL_BREADTH_RIGHT = np.float32([-20.0, -56.1, 10.0]) #17
P3D_FRONTAL_BREADTH_LEFT = np.float32([-20.0, 56.1, 10.0]) #26
P3D_SELLION = np.float32([0.0, 0.0, 0.0]) #27
P3D_NOSE = np.float32([21.1, 0.0, -48.0]) #30
P3D_SUB_NOSE = np.float32([5.0, 0.0, -52.0]) #33
P3D_RIGHT_EYE = np.float32([-20.0, -65.5,-5.0]) #36
P3D_RIGHT_TEAR = np.float32([-10.0, -40.5,-5.0]) #39
P3D_LEFT_TEAR = np.float32([-10.0, 40.5,-5.0]) #42
P3D_LEFT_EYE = np.float32([-20.0, 65.5,-5.0]) #45
#P3D_LIP_RIGHT = np.float32([-20.0, 65.5,-5.0]) #48
#P3D_LIP_LEFT = np.float32([-20.0, 65.5,-5.0]) #54
P3D_STOMION = np.float32([10.0, 0.0, -75.0]) #62

#The points to track
#These points are the ones used by PnP
# to estimate the 3D pose of the face
TRACKED_POINTS = (0, 4, 8, 12, 16, 17, 26, 27, 30, 33, 36, 39, 42, 45, 62)
ALL_POINTS = list(range(0,68)) #Used for debug only


def main():

    #Defining the video capture object
    video_capture = cv2.VideoCapture(0)

    if(video_capture.isOpened() == False):
        print("Error: the resource is busy or unvailable")
    else:
        print("The video source has been opened correctly...")

    #Create the main window and move it
    cv2.namedWindow('Video')
    cv2.moveWindow('Video', 20, 20)

    #Obtaining the CAM dimension
    cam_w = int(video_capture.get(3))
    cam_h = int(video_capture.get(4))

    #Defining the camera matrix.
    #To have better result it is necessary to find the focal
    # lenght of the camera. fx/fy are the focal lengths (in pixels)
    # and cx/cy are the optical centres. These values can be obtained
    # roughly by approximation, for example in a 640x480 camera:
    # cx = 640/2 = 320
    # cy = 480/2 = 240
    # fx = fy = cx/tan(60/2 * pi / 180) = 554.26
    c_x = cam_w / 2
    c_y = cam_h / 2
    f_x = c_x / np.tan(60/2 * np.pi / 180)
    f_y = f_x

    #Estimated camera matrix values.
    camera_matrix = np.float32([[f_x, 0.0, c_x],
                                   [0.0, f_y, c_y],
                                   [0.0, 0.0, 1.0] ])

    #print("Estimated camera matrix: \n" + str(camera_matrix) + "\n")

    #These are the camera matrix values estimated on my webcam with
    # the calibration code (see: src/calibration):
    camera_matrix = np.float32([[602.10618226,          0.0, 320.27333589],
                                   [         0.0, 603.55869786,  229.7537026],
                                   [         0.0,          0.0,          1.0] ])

    #Distortion coefficients
    #camera_distortion = np.float32([0.0, 0.0, 0.0, 0.0, 0.0])

    #Distortion coefficients estimated by calibration
    camera_distortion = np.float32([ 0.06232237, -0.41559805,  0.00125389, -0.00402566,  0.04879263])


    #This matrix contains the 3D points of the
    # 11 landmarks we want to find. It has been
    # obtained from antrophometric measurement
    # on the human head.
    landmarks_3D = np.float32([P3D_RIGHT_SIDE,
                                  P3D_GONION_RIGHT,
                                  P3D_MENTON,
                                  P3D_GONION_LEFT,
                                  P3D_LEFT_SIDE,
                                  P3D_FRONTAL_BREADTH_RIGHT,
                                  P3D_FRONTAL_BREADTH_LEFT,
                                  P3D_SELLION,
                                  P3D_NOSE,
                                  P3D_SUB_NOSE,
                                  P3D_RIGHT_EYE,
                                  P3D_RIGHT_TEAR,
                                  P3D_LEFT_TEAR,
                                  P3D_LEFT_EYE,
                                  P3D_STOMION])

    #Declaring the two classifiers
    my_cascade = haarCascade("/Users/lamh/Documents/deepgaze/etc/xml/haarcascade_frontalface_alt.xml", "/Users/lamh/Documents/deepgaze/etc/xml/haarcascade_profileface.xml")
    #TODO If missing, example file can be retrieved from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
    my_detector = faceLandmarkDetection('/Users/lamh/Documents/deepgaze/etc/shape_predictor_68_face_landmarks.dat')

    #Error counter definition
    no_face_counter = 0

    #Variables that identify the face
    #position in the main frame.
    face_x1 = 0
    face_y1 = 0
    face_x2 = 0
    face_y2 = 0
    face_w = 0
    face_h = 0

    #Variables that identify the ROI
    #position in the main frame.
    roi_x1 = 0
    roi_y1 = 0
    roi_x2 = cam_w
    roi_y2 = cam_h
    roi_w = cam_w
    roi_h = cam_h
    roi_resize_w = int(cam_w/10)
    roi_resize_h = int(cam_h/10)

    try:
        while(True):

            # Capture frame-by-frame
            ret, frame = video_capture.read()
            gray = cv2.cvtColor(frame[roi_y1:roi_y2, roi_x1:roi_x2], cv2.COLOR_BGR2GRAY)

            #Return code: 1=Frontal, 2=FrontRotLeft,
            # 3=FrontRotRight, 4=ProfileLeft, 5=ProfileRight.
            my_cascade.findFace(gray, True, True, True, True, 1.10, 1.10, 1.15, 1.15, 40, 40, rotationAngleCCW=30, rotationAngleCW=-30, lastFaceType=my_cascade.face_type)

            #Accumulate error values in a counter
            if(my_cascade.face_type == 0):
                no_face_counter += 1

            #If any face is found for a certain
            #number of cycles, then the ROI is reset
            if(no_face_counter == 50):
                no_face_counter = 0
                roi_x1 = 0
                roi_y1 = 0
                roi_x2 = cam_w
                roi_y2 = cam_h
                roi_w = cam_w
                roi_h = cam_h

            #Checking which kind of face it is returned
            if(my_cascade.face_type > 0):

                #Face found, reset the error counter
                no_face_counter = 0

                #Because the dlib landmark detector wants a precise
                #boundary box of the face, it is necessary to resize
                #the box returned by the OpenCV haar detector.
                #Adjusting the frame for profile left
                if(my_cascade.face_type == 4):
                    face_margin_x1 = 20 - 10 #resize_rate + shift_rate
                    face_margin_y1 = 20 + 5 #resize_rate + shift_rate
                    face_margin_x2 = -20 - 10 #resize_rate + shift_rate
                    face_margin_y2 = -20 + 5 #resize_rate + shift_rate
                    face_margin_h = -0.7 #resize_factor
                    face_margin_w = -0.7 #resize_factor
                #Adjusting the frame for profile right
                elif(my_cascade.face_type == 5):
                    face_margin_x1 = 20 + 10
                    face_margin_y1 = 20 + 5
                    face_margin_x2 = -20 + 10
                    face_margin_y2 = -20 + 5
                    face_margin_h = -0.7
                    face_margin_w = -0.7
                #No adjustments
                else:
                    face_margin_x1 = 0
                    face_margin_y1 = 0
                    face_margin_x2 = 0
                    face_margin_y2 = 0
                    face_margin_h = 0
                    face_margin_w = 0

                #Updating the face position
                face_x1 = my_cascade.face_x + roi_x1 + face_margin_x1
                face_y1 = my_cascade.face_y + roi_y1 + face_margin_y1
                face_x2 = my_cascade.face_x + my_cascade.face_w + roi_x1 + face_margin_x2
                face_y2 = my_cascade.face_y + my_cascade.face_h + roi_y1 + face_margin_y2
                face_w = my_cascade.face_w + int(my_cascade.face_w * face_margin_w)
                face_h = my_cascade.face_h + int(my_cascade.face_h * face_margin_h)

                #Updating the ROI position
                roi_x1 = face_x1 - roi_resize_w
                if (roi_x1 < 0): roi_x1 = 0
                roi_y1 = face_y1 - roi_resize_h
                if(roi_y1 < 0): roi_y1 = 0
                roi_w = face_w + roi_resize_w + roi_resize_w
                if(roi_w > cam_w): roi_w = cam_w
                roi_h = face_h + roi_resize_h + roi_resize_h
                if(roi_h > cam_h): roi_h = cam_h
                roi_x2 = face_x2 + roi_resize_w
                if (roi_x2 > cam_w): roi_x2 = cam_w
                roi_y2 = face_y2 + roi_resize_h
                if(roi_y2 > cam_h): roi_y2 = cam_h

                #Debugging printing utilities
                #if(DEBUG == True):
                if(DEBUG == False):
                    #print("FACE: ", face_x1, face_y1, face_x2, face_y2, face_w, face_h)
                    #print("ROI: ", roi_x1, roi_y1, roi_x2, roi_y2, roi_w, roi_h)
                    #Drawing a green rectangle
                    # (and text) around the face.
                    text_x1 = face_x1
                    text_y1 = face_y1 - 3
                    if(text_y1 < 0): text_y1 = 0
                    cv2.putText(frame, "FACE", (text_x1,text_y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1);
                    cv2.rectangle(frame,
                                 (face_x1, face_y1),
                                 (face_x2, face_y2),
                                 (0, 255, 0),
                                  2)

                # In case of a frontal/rotated face it
                # is called the landamark detector
                if(my_cascade.face_type > 0):
                    landmarks_2D = my_detector.returnLandmarks(frame, face_x1, face_y1, face_x2, face_y2, points_to_return=TRACKED_POINTS)

                    if(DEBUG == True):
                        #cv2.drawKeypoints(frame, landmarks_2D)

                        for point in landmarks_2D:
                            cv2.circle(frame,( point[0], point[1] ), 2, (0,0,255), -1)


                    # Applying the PnP solver to find the 3D pose
                    # of the head from the 2D position of the
                    # landmarks.
                    # retval - bool
                    # rvec - Output rotation vector that, together with tvec, brings
                    # points from the model coordinate system to the camera coordinate system.
                    # tvec - Output translation vector.
                    retval, rvec, tvec = cv2.solvePnP(landmarks_3D,
                                                      landmarks_2D,
                                                      camera_matrix, camera_distortion)

                    # Now we project the 3D points into the image plane
                    # Creating a 3-axis to be used as reference in the image.
                    axis = np.float32([[50,0,0],
                                          [0,50,0],
                                          [0,0,50]])
                    imgpts, jac = cv2.projectPoints(axis, rvec, tvec, camera_matrix, camera_distortion)

                    # Drawing the three axis on the image frame.
                    selection_xy = (landmarks_2D[7][0], landmarks_2D[7][1])
                    print("*** Position: {0}, pitch, yaw, roll: {1}, {2}, {3} ***".format(selection_xy, np.degrees(rvec)[0], np.degrees(rvec)[1], np.degrees(rvec)[2]))
                    if(max(imgpts[1].ravel()) <= 500000 and min(imgpts[1].ravel()) >= -500000):
                        cv2.line(frame, selection_xy, tuple(imgpts[1].ravel()), (0,255,0), 3) #GREEN

                    if(max(imgpts[2].ravel()) <= 500000 and min(imgpts[2].ravel()) >= -500000):
                        cv2.line(frame, selection_xy, tuple(imgpts[2].ravel()), (255,0,0), 3) #BLUE

                    if(max(imgpts[0].ravel()) <= 500000 and min(imgpts[0].ravel()) >= -500000):
                        cv2.line(frame, selection_xy, tuple(imgpts[0].ravel()), (0,0,255), 3) #RED

            #Drawing a yellow rectangle
            # (and text) around the ROI.
            if(DEBUG == True):
                text_x1 = roi_x1
                text_y1 = roi_y1 - 3
                if(text_y1 < 0): text_y1 = 0
                #cv2.putText(frame, "ROI", (text_x1,text_y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1);
                #cv2.rectangle(frame,
                #             (roi_x1, roi_y1),
                #             (roi_x2, roi_y2),
                #             (0, 255, 255),
                #             2)

            #Showing the frame and waiting
            # for the exit command
            cv2.imshow('Video', frame)
            key = cv2.waitKey(1)
            if key == 27:
                cv2.destroyAllWindowns()
                break
            #if cv2.waitKey(1) & 0xFF == ord('q'): break
    except KeyboardInterrupt:
        #Release the camera
        video_capture.release()
        print("KeyboardInterrupt: camera released. Bye ...")


if __name__ == "__main__":
    main()
