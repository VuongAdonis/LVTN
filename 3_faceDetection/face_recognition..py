import cv2
import face_recognition as FR

font = cv2.FONT_HERSHEY_SIMPLEX
donFace = FR.load_image_file(
    'C:\\Users\\Acer\\Desktop\\AI_for_LVTN\\3_faceDetection\\demoImages\\known\\Donald Trump.jpg')
faceLoc = FR.face_locations(donFace)[0]
donFaceEncode = FR.face_encodings(donFace)

nancyFace = FR.load_image_file(
    'C:\\Users\\Acer\\Desktop\\AI_for_LVTN\\3_faceDetection\\demoImages\\known\\Nancy Pelosi.jpg')
faceLoc = FR.face_locations(nancyFace)[0]
nancyFaceEncode = FR.face_encodings(nancyFace)

knownEncodings = [donFaceEncode, nancyFaceEncode]
names = ['Donal Trump', 'Nancy Pelosi']

unknownFace = FR.load(
    'C:/Users/Acer/Desktop/AI_for_LVTN/3_faceDetection/demoImages/unknown/u1.jpg')
unknownFaceBGR = cv2.cvtColor(unknownFace, cv2.COLOR_RGB2BGR)
faceLocations = FR.face_locations(unknownFace)
unknownEncodings = FR.face_encodings(unknownFace, faceLocations)

for faceLocation, unknownEncoding in zip(faceLocations, unknownEncodings):
    top, right, bottom, left = faceLocation
    print(faceLocation)
    cv2.rectangle(unknownFaceBGR, (left, top), (right, bottom), (255, 0, 0), 3)
    name = 'Unknown Person'
    matches = FR.compare_faces(knownEncodings, unknownEncoding)
    print(matches)

cv2.waitKey(5000)
