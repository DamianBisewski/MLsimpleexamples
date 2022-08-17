import cv2
import imutils
import numpy

def zad1(filename):
    original = cv2.imread(filename)
    img = imutils.resize(original.copy(), width=500)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow('img', img)
    cv2.waitKey()
    cv2.destroyAllWindows()


def zad2(filename):
    original = cv2.imread(filename)
    img = imutils.resize(original.copy(), width=500)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    eyes = eye_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in eyes:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    smiles = smile_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in smiles:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

    cv2.imshow('img', img)
    cv2.waitKey()
    cv2.destroyAllWindows()

    print(f"Znaleziono {len(faces)} twarze")


def zad3(filename):
    cap = cv2.VideoCapture(filename)
    if(cap.isOpened() == False):
        print("Cannot show movie")

    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            image = frame.copy()
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            for (x, y, w, h) in faces:
                cv2.rectangle(gray, (x, y), (x + w, y + h), (255, 0, 0), 2)

            cv2.imshow('Vid', imutils.resize(gray, width=500))

        if cv2.waitKey(1) and 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def zad4(filename):
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    cv2.startWindowThread()
    cap = cv2.VideoCapture(filename)
    if(cap.isOpened() == False):
        print("Cannot show movie")
    while(cap.isOpened()):
        ret, frame = cap.read()
        frame = cv2.resize(frame, (800, 560))
        gray_filter = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        boxes, weights = hog.detectMultiScale(frame, winStride=(8,8))
        boxes = numpy.array([[x, y, x+w, y+h] for (x,y,w,h) in boxes])
        for(xa, ya, xb, yb) in boxes:
            cv2.rectangle(frame, (xa, ya), (xb, yb), (0,255,0), 1)
        cv2.imshow("Video", frame)
        if cv2.waitKey(1) and 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    zad1()          #Here insert a photo with a face
    zad2()          #Here insert a photo with many faces
    zad3()          #Here insert a movie with your face
    zad4()          #Here insert a movie with people walking, for example on a street
