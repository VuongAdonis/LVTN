import cv2
import face_recognition as FR
import os

"""
Note: Numpy > 2.x.x will raise error when use face_recognition
So you need to download numpy 1.x.x
"""

font = cv2.FONT_HERSHEY_SIMPLEX
width = 640
height = 360
cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
cam.set(cv2.CAP_PROP_FPS, 30)
cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

imgDir = 'C:/Users/Acer/Desktop/AI_for_LVTN/3_faceDetection/demoImages/liveWebcam'
knownEncodings = []
names = []

for root, dirs, files in os.walk(imgDir):
    for file in files:
        fullFilePath = os.path.join(root, file)
        name = os.path.splitext(file)[0]
        knownFace = FR.load_image_file(fullFilePath)
        knownFaceEncode = FR.face_encodings(knownFace)[0]

        knownEncodings.append(knownFaceEncode)
        names.append(name)

while True:
    ignore, unknownFace = cam.read()

    unknownFaceRGB = cv2.cvtColor(unknownFace, cv2.COLOR_BGR2RGB)
    faceLocations = FR.face_locations(unknownFaceRGB)
    unknownEncodings = FR.face_encodings(unknownFaceRGB, faceLocations)

    for faceLocation, unknownEncoding in zip(faceLocations, unknownEncodings):
        top, right, bottom, left = faceLocation
        print(faceLocation)
        cv2.rectangle(unknownFace, (left, top),
                      (right, bottom), (255, 0, 0), 3)
        name = 'Unknown Person'
        matches = FR.compare_faces(knownEncodings, unknownEncoding)
        print(matches)
        if True in matches:
            matchIndex = matches.index(True)
            print(matchIndex)
            print(names[matchIndex])
            name = names[matchIndex]
        cv2.putText(unknownFace, name, (left, top), font, 0.75, (0, 0, 255), 2)

    cv2.imshow('my faces', unknownFace)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
