import cv2
import numpy as np
import pickle

# import os
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

print(cv2.__version__)


class mpHands:
    import mediapipe as mp

    def __init__(self, maxHands=2, tol1=0.5, modelComplexity=1, tol2=0.5):
        self.tol1 = tol1
        self.tol2 = tol2
        self.maxHands = maxHands
        self.modelComplexity = modelComplexity
        self.hands = self.mp.solutions.hands.Hands(
            False, self.maxHands, self.modelComplexity, self.tol1, self.tol2)

    def Marks(self, frame):
        myHands = []
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frameRGB)
        if results.multi_hand_landmarks != None:
            for handLandMarks in results.multi_hand_landmarks:
                myHand = []
                for landMark in handLandMarks.landmark:
                    myHand.append(
                        (int(landMark.x*width), int(landMark.y*height)))
                myHands.append(myHand)
        return myHands


def findDistances(handData):
    distMatrix = np.zeros([len(handData), len(handData)], dtype='float')
    palmSize = ((handData[0][0]-handData[9][0])
                ** 2 + (handData[0][1]-handData[9][1])**2)**(1./2.)
    for row in range(0, len(handData)):
        for col in range(0, len(handData)):
            distMatrix[row][col] = (((handData[row][0]-handData[col][0])
                                    ** 2 + (handData[row][1]-handData[col][1])**2)**(1./2.))/palmSize
    return distMatrix


def findError(gestureMatrix, unknownMatrix, keyPoints):
    error = 0
    for row in keyPoints:
        for col in keyPoints:
            error = error+abs(gestureMatrix[row][col]-unknownMatrix[row][col])
    return error


def findGesture(unknownGesture, knownGestures, keyPoints, gestNames, tol):
    errorArray = []
    for i in range(0, len(gestNames), 1):
        error = findError(knownGestures[i], unknownGesture, keyPoints)
        errorArray.append(error)
    errorMin = errorArray[0]
    minIndex = 0
    for i in range(0, len(errorArray), 1):
        if errorArray[i] < errorMin:
            errorMin = errorArray[i]
            minIndex = i
    if errorMin < tol:
        gesture = gestNames[minIndex]
    if errorMin >= tol:
        gesture = 'Unknown'
    return gesture


width = 1280
height = 720
cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
cam.set(cv2.CAP_PROP_FPS, 30)
cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
findHands = mpHands(1)

keyPoints = [0, 4, 5, 9, 13, 17, 8, 12, 16, 20]
train = int(input('Enter 1 to Train, Enter 0 to Recognize: '))
if train == 1:
    trainCnt = 0
    knownGestures = []
    numGest = int(input('How many Gestures do you want? '))
    gestNames = []
    for i in range(0, numGest, 1):
        promt = 'Name of Gesture #' + str(i+1) + ' '
        name = input(promt)
        gestNames.append(name)
    print(gestNames)
    trainName = input('FileName for training data? (Press Enter for Default) ')
    if trainName == '':
        trainName = 'default'
    trainName = './4_gestureDetection/' + trainName+'.pkl'
if train == 0:
    trainName = input(
        'What training data do you want to use? (Press Enter for Default) ')
    if trainName == '':
        trainName = 'default'
    trainName = './4_gestureDetection/' + trainName + '.pkl'
    with open(trainName, 'rb') as f:
        gestNames = pickle.load(f)
        knownGestures = pickle.load(f)

tol = 10

while True:
    ignore,  frame = cam.read()
    frame = cv2.resize(frame, (width, height))
    handData = findHands.Marks(frame)
    if train == 1:
        if handData != []:
            print('Please show gesture ',
                  gestNames[trainCnt], ': Press t when Ready')
            if cv2.waitKey(1) & 0xff == ord('t'):
                knownGesture = findDistances(handData[0])
                knownGestures.append(knownGesture)
                trainCnt = trainCnt+1
                if trainCnt == numGest:
                    train = 0
                    with open(trainName, 'wb') as f:
                        pickle.dump(gestNames, f)
                        pickle.dump(knownGestures, f)

    if train == 0:
        if handData != []:
            unknownGesture = findDistances(handData[0])
            # error = findError(knownGesture, unknownGesture, keyPoints)
            myGesture = findGesture(
                unknownGesture, knownGestures, keyPoints, gestNames, tol)
            cv2.putText(frame, myGesture, (100, 175),
                        cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 8)

    for hand in handData:
        for ind in keyPoints:
            cv2.circle(frame, hand[ind], 25, (255, 0, 255), 3)
    cv2.imshow('my WEBcam', frame)
    cv2.moveWindow('my WEBcam', 0, 0)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
cam.release()
