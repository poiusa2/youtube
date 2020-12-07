import cv2
import numpy as np
from os import makedirs
from os.path import isdir

face_dirs = 'faces/'
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


def face_extractor(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,1.3,5)
    if faces is():
        return None #얼굴이 없으면 패스
    for(x,y,w,h) in faces:
        cropped_face = img[y:y+h, x:x+w]
    return cropped_face #얼굴부위만 이미지로하고 return

def take_pictures(name):
    if not isdir(face_dirs+name):
        makedirs(face_dirs+name)
    cap = cv2.VideoCapture(0)
    count = 0
    while True:
        ret, frame = cap.read()
        if face_extractor(frame) is not None:
            count+=1
            face = cv2.resize(face_extractor(frame),(200,200)) #200*200사이즈로 규격
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY) #흑백으로 바꿈
            file_name_path = face_dirs + name+'/user'+str(count)+'.jpg'
            cv2.imwrite(file_name_path,face)
            cv2.putText(face,str(count),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
            cv2.imshow('Face Cropper',face)
        else:
            print("Face not Found")
            pass
        if cv2.waitKey(1)==13 or count==200: #enter키를 누르거나 얼굴사진 200장 얻으면 종료
            break
    cap.release()
    cv2.destroyAllWindows()
    print('Colleting Samples Complete!!!')

if __name__ == "__main__":
    # 사진 저장할 이름을 넣어서 함수 호출
    take_pictures('teacher')


    # 위의 같이 take_pictures('jin') 함수를 호출하면 faces/jin 폴더가 생성되고
    # 이곳에 200장의 사진을 카메라로 찍어 저장하게 된다.
    # 이걸 만약 take_pictures('teacher')이라고 바꿔 실행하면 faces/teacher 이라는 폴더가 생성되고
    # 마찬가지로 200장의 사진이 찍히는 것이다.

