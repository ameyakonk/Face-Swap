import cv2
import numpy as np
import dlib

class FacePoints():

# This below mehtod will draw all those points which are from 0 to 67 on face one by one.
    def drawPoints(self, image, faceLandmarks, startpoint, endpoint, isClosed=False):
        points = []
        for i in range(startpoint, endpoint+1):
            point = [faceLandmarks.part(i).x, faceLandmarks.part(i).y]
            points.append(point)

        points = np.array(points, dtype=np.int32)
        cv2.polylines(image, [points], isClosed, (255, 200, 0), thickness=2, lineType=cv2.LINE_8)

    # Use this function for 70-points facial landmark detector model
    # we are checking if points are exactly equal to 68, then we draw all those points on face one by one
    def facePoints(self, image, faceLandmarks):
        assert(faceLandmarks.num_parts == 68)
        self.drawPoints(image, faceLandmarks, 0, 16)           # Jaw line
        self.drawPoints(image, faceLandmarks, 17, 21)          # Left eyebrow
        self.drawPoints(image, faceLandmarks, 22, 26)          # Right eyebrow
        self.drawPoints(image, faceLandmarks, 27, 30)          # Nose bridge
        self.drawPoints(image, faceLandmarks, 30, 35, True)    # Lower nose
        self.drawPoints(image, faceLandmarks, 36, 41, True)    # Left eye
        self.drawPoints(image, faceLandmarks, 42, 47, True)    # Right Eye
        self.drawPoints(image, faceLandmarks, 48, 59, True)    # Outer lip
        self.drawPoints(image, faceLandmarks, 60, 67, True)    # Inner lip

    # Use this function for any model other than

class FaceFeatureExtractor():
    def writeFaceLandmarksToLocalFile(self, faceLandmarks, fileName):
      with open(fileName, 'w') as f:
        for p in faceLandmarks.parts():
            f.write("%s %s\n" %(int(p.x),int(p.y)))

        f.close()

    def execute(self, image):
    # location of the model (path of the model).
        Model_PATH = "shape_predictor_68_face_landmarks.dat"

        # now from the dlib we are extracting the method get_frontal_face_detector()
        # and assign that object result to frontalFaceDetector to detect face from the image with 
        # the help of the 68_face_landmarks.dat model
        frontalFaceDetector = dlib.get_frontal_face_detector()

        # Now the dlip shape_predictor class will take model and with the help of that, it will show 
        faceLandmarkDetector = dlib.shape_predictor(Model_PATH)
    
        imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Now this line will try to detect all faces in an image either 1 or 2 or more faces
        allFaces = frontalFaceDetector(imageRGB, 0)


        print("List of all faces detected: ",len(allFaces))

        # List to store landmarks of all detected faces
        allFacesLandmark = []
        
        # Below loop we will use to detect all faces one by one and apply landmarks on them
        total_points = []
        for k in range(0, len(allFaces)):
            points = []
            # dlib rectangle class will detecting face so that landmark can apply inside of that area
            faceRectangleDlib = dlib.rectangle(int(allFaces[k].left()),int(allFaces[k].top()),
                int(allFaces[k].right()),int(allFaces[k].bottom()))

            # Now we are running loop on every detected face and putting landmark on that with the help of faceLandmarkDetector
            detectedLandmarks = faceLandmarkDetector(imageRGB, faceRectangleDlib)
            
            # count number of landmarks we actually detected on image
            if k==0:
                print("Total number of face landmarks detected ",len(detectedLandmarks.parts()))

            # Svaing the landmark one by one to the output folder
            allFacesLandmark.append(detectedLandmarks)

            for p in detectedLandmarks.parts():
                points.append((int(p.x), int(p.y)))

            total_points.append(points)
        

        return total_points