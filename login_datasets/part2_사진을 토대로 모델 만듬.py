import cv2
import numpy as np
from os import listdir
from os.path import isdir, isfile, join
import time
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
def train(name):
    data_path = 'faces/' + name + '/'
    #파일만 리스트로만듬
    face_pics = [f for f in listdir(data_path) if isfile(join(data_path,f))]
    Training_Data, Labels = [], []
    for i, files in enumerate(face_pics):
        image_path = data_path + face_pics[i]
        images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if images is None: # 확장자가 jpg가 아닌 경우 무시
            continue
        Training_Data.append(np.asarray(images, dtype=np.uint8))
        Labels.append(i)

    if len(Labels) ==0:
        print("There is no data")
        return None
    Labels = np.asarray(Labels, dtype=np.int32) #사진 200개
    model = cv2.face.LBPHFaceRecognizer_create()
    model.train(np.asarray(Training_Data), np.asarray(Labels)) #Matrix만들기
    model.save("./model/{0}.yml".format(name))
    print("Model Training Complete!!!!!")
    return model  #학습 모델 리턴

#여러 사용자 학습
def trains():
    data_path = 'faces/'
    model_dirs = [f for f in listdir(data_path) if isdir(join(data_path, f))]
    print(model_dirs)
    # 학습 모델 저장할 딕셔너리
    models = {}
    # 각 폴더에 있는 얼굴들 학습
    for model in model_dirs:
        print('start :' + model)
        # 학습 시작
        result = train(model)
        # 학습이 안되었다면 패스!
        if result is None:
            continue
        # 학습되었으면 저장
        print('end :' + model)
        models[model] = result
    # 학습된 모델 딕셔너리 리턴
    print('dic:',models)
    return models

#한꺼번에 여러 사람이 수집된 폴더로 여러 모델을 만들거면 trains를 실행
#models = trains()

#한 사람을 추가로 하려면 train을 실행
#train("teacher") #"이름"칸에 있는 건 그 이름의 사람이여야 함
