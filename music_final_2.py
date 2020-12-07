import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic
from PyQt5.QtCore import *
from selenium import webdriver
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences
from os import listdir
import time
import cv2
import numpy as np
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import pandas as pd
import random
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup


form_window = uic.loadUiType('./music_base_3.ui')[0]
face_detection = cv2.CascadeClassifier('./login_datasets/haarcascade_frontalface_default.xml')
emotion_classifier = load_model('./login_datasets/emotion_model.hdf5', compile=False)

class Worker(QThread):
    playlist_changed = pyqtSignal(str)

    def __init__(self, parent):
        super(Worker, self).__init__(parent)
        self.parent = parent
        self.title = '플레이리스트를 찾는 중입니다'

    # def __del__(self):
    #     if self.working == False:
    #         #self.wait()
    #         print('___End Thread ___')
    #         self.driver.close()
    #     else:
    #         pass

    def run(self):
        try:
            self.driver.close()
        except:
            pass
        options = webdriver.ChromeOptions()
        #options.add_argument('headless')
        options.add_argument('disable-gpu')
        options.add_argument('lang=ko_KR')
        self.driver = webdriver.Chrome('./chromedriver', options=options)
        self.driver.implicitly_wait(0.01)
        try:
            self.youtube_play()
        except:
            pass

    def youtube_play(self):
        if self.parent.emotion_word:
            emotion_word = self.parent.emotion_word
        else:
            emotion_word = ''
        if self.parent.today_word:
            today_word = self.parent.today_word
        else:
            today_word = ''
        if self.parent.genre_word:
            genre_word = self.parent.genre_word[0]
        else:
            genre_word = ''
        playlist = ['플레이리스트', 'playlist','노래 재생목록']
        playlist_word = random.sample(playlist,1)[0]
        print('0')

        url = 'https://www.youtube.com/results?search_query={}'.format(today_word+' '+emotion_word+' '+genre_word+playlist_word)
        print('1')
        time.sleep(1)
        self.driver.get(url)
        print('2')
        time.sleep(0.1)
        self.driver.find_element_by_xpath(
            f'/html/body/ytd-app/div/ytd-page-manager/ytd-search/div[1]/ytd-two-column-search-results-renderer/div/ytd-section-list-renderer/div[2]/ytd-item-section-renderer/div[3]/ytd-video-renderer[{random.randint(1, 4)}]/div[1]/ytd-thumbnail').click()
        try: #광고 스킵 넣을 거임
            self.driver.find_element_by_xpath("//*[@class='ytp-large-play-button ytp-button']")
        except:
            pass
        self.body = self.driver.find_element_by_tag_name('body')
        time.sleep(0.5)
        what = '//*[@id="container"]/h1/yt-formatted-string'
        self.title = str(self.driver.find_element_by_xpath(what).text)


#로그인 창
class Login(QDialog):
    def __init__(self,parent):
        super(Login, self).__init__(parent)
        self.parent = parent
        option_ui = './login_try.ui'
        uic.loadUi(option_ui, self)
        self.initUI()
        self.setWindowTitle("로그인")
        self.exec_()

    def initUI(self):
        self.id= None
        self.password = None
        self.name = None
        #self.face_classifier = cv2.CascadeClassifier('./login_datasets/haarcascade_frontalface_default.xml')
        self.data_path = './login_datasets/model/'
        self.btn_login.accepted.connect(self.btn_login_clicked)
        self.btn_login.rejected.connect(exit)
        self.btn_face.clicked.connect(self.face_login)

    def btn_login_clicked(self):
        self.id = str(self.le_id.text())
        self.password = str(self.le_password.text())
        if self.id in list(self.parent.df_customer.ID):
            idx = self.parent.df_customer[self.parent.df_customer['ID'] == self.id].index[0]
            if self.password == self.parent.df_customer['PASSWORD'][idx]:
                self.parent.login_flag = True
            else:
                self.le_id.clear()
                self.le_password.clear()
                self.parent.login_flag = False
                QMessageBox.warning(self, 'ERROR', '비밀번호가 틀립니다.')
        else:
            self.le_id.clear()
            self.le_password.clear()
            self.parent.login_flag = False
            QMessageBox.warning(self, 'ERROR', '비밀번호가 틀리거나 없는 아이디입니다.')

    def face_login(self):
        # 얼굴 검출
        #self.login_flag = False
        self.id = None
        def face_detector(img, size=0.5):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_detection.detectMultiScale(gray, 1.3, 5)
            if faces is ():
                return img, []
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
                roi = img[y:y + h, x:x + w]
                roi = cv2.resize(roi, (200, 200))
            return img, roi  # 검출된 좌표에 사각 박스 그리고(img), 검출된 부위를 잘라(roi) 전달
        # 인식 시작
        def run(models):
            cap = cv2.VideoCapture(0)  # 카메라 열기
            capture_duration = 15  # 시간
            start_time = time.time()
            while (int(time.time() - start_time) < capture_duration):  # 시간안에만 작동
                ret, frame = cap.read()  # 카메라로 부터 사진 한장 읽기
                image, face = face_detector(frame)  # 얼굴 검출 시도
                try:
                    min_score = 999  # 가장 낮은 점수로 예측된 사람의 점수 #999 -> 99
                    id = ""  # 가장 높은 점수로 예측된 사람의 이름
                    # 검출된 사진을 흑백으로 변환
                    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                    # 위에서 학습한 모델로 예측시도
                    for key, model in models.items():  # 딕셔너리라서 key:model == jin:model명
                        result = model.predict(face)  # (45,65.112456) 이런식으로 출력된다
                        if min_score > result[1]:
                            min_score = result[1]
                            id = key
                    # min_score 신뢰도이고 0에 가까울수록 자신과 같다는 뜻이다. minscore의 점수가 갱신됨
                    if min_score < 500:
                        confidence = int(100 * (1 - (min_score) / 300))
                    # 77 보다 크면 동일 인물로 간주해 UnLocked!
                    if confidence > 77:
                        cv2.putText(image, "Unlocked : " + id, (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                        cv2.imshow('Face Cropper', image)
                        self.id = id
                        break
                    else:
                        # 75 이하면 타인.. Locked!!!
                        cv2.putText(image, "Locked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                        cv2.imshow('Face Cropper', image)
                except:
                    # 얼굴 검출 안됨
                    cv2.putText(image, "Face Not Found", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
                    cv2.imshow('Face Cropper', image)
                    pass
                if cv2.waitKey(1) == 13:
                    break
            cap.release()
            cv2.destroyAllWindows()
            if not self.id == None:
                self.parent.login_flag = True
                print('flag:',self.parent.login_flag)
                print('얼굴로 로그인 성공')
            else:
                self.parent.login_flag = False
                pass


        model_dirs = [f[:-4] for f in listdir(self.data_path) if f.endswith(".yml")]  # yml파일 이름만 불러온다
        models = {}
        for model_name in model_dirs:
            print('start :' + model_name)
            model = cv2.face.LBPHFaceRecognizer_create()
            model.read("./login_datasets/model/{}.yml".format(model_name))
            models[model_name] = model
        run(models)

    def closeEvent(self, QCloseEvent):
        ans = QMessageBox.question(self, '종료하기', '종료하시겠습니까?', QMessageBox.Yes | QMessageBox.No,
                                   QMessageBox.Yes)
        if ans == QMessageBox.Yes:
            sys.exit()
        else:
            QCloseEvent.ignore()







#본체
class Exam(QWidget, form_window):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle('YOUTUBE MUSIC')
        self.initUI()
        #self.working = False

    def initUI(self):
        self.df_customer = pd.read_excel('./login_datasets/customer.xlsx')
        self.emotion_df = pd.read_excel('./emotion_final.xlsx')
        self.setStyleSheet("background-color: #ffffff")
        self.dialog_login()
        self.btn_logout.clicked.connect(self.dialog_login)
        self.btn_emotion.clicked.connect(self.emotion_read)
        self.cb_feeling.addItem("Happy")
        self.cb_feeling.addItem("Sad")
        self.cb_feeling.addItem("Angry")
        self.cb_feeling.addItem("Disgusting")
        self.cb_feeling.addItem("Fearful")
        self.cb_feeling.addItem("Surprising")
        self.cb_feeling.addItem("Neutral")
        self.cb_feeling.currentIndexChanged.connect(self.cb_feeling_slot)
        self.classic.stateChanged.connect(self.genre_slot)
        self.bgm.stateChanged.connect(self.genre_slot)
        self.jazz.stateChanged.connect(self.genre_slot)
        self.k_ballard.stateChanged.connect(self.genre_slot)
        self.k_hiphop.stateChanged.connect(self.genre_slot)
        self.k_pop.stateChanged.connect(self.genre_slot)
        self.k_trot.stateChanged.connect(self.genre_slot)
        self.btn_ok.clicked.connect(self.youtube_play_slot)
        self.EMOTIONS = ["Angry", "Disgusting", "Fearful", "Happy", "Sad", "Surprising", "Neutral"]
        self.EMOTIONS_KOR = {'Angry': '화나', 'Disgusting': '기분 안 좋아', 'Fearful': '걱정스러워', 'Happy': '행복해', 'Sad': '우울해',
                        'Surprising': '놀라', 'Neutral': '그냥 그래'}
        self.worker = Worker(self)
        self.btn_allstop.clicked.connect(self.youtube_allstop_slot)
        self.btn_stop.clicked.connect(self.youtube_stop_slot)
        self.genre_word = []

    def cb_feeling_slot(self):
        feeling = self.cb_feeling.currentText()
        self.lbl_feeling.setText(self.EMOTIONS_KOR[feeling])
        df = self.emotion_df[self.emotion_df['feeling'] == feeling]
        self.emotion_word = df['word'].sample().iloc[0]
        print('감정수동으로 emotion_word: ',self.emotion_word)

    def dialog_login(self):
        self.log_in = Login(self)  # 로그인창 발동
        self.login_flag == False
        while True:
            if self.login_flag == True:
                self.id = self.log_in.id
                self.idx = self.df_customer[self.df_customer['ID'] == self.id].index[0]
                self.name = self.df_customer['NAME'][self.idx]
                self.lbl_name.setText(self.name)
                break
            else:
                self.log_in = Login(self)
                print('로그인실패')


    def emotion_read(self):
        cap = cv2.VideoCapture(0)  # 카메라 열기
        count = 0
        pre = np.array([0, 0, 0, 0, 0, 0, 0])
        while count < 30:
            count += 1
            ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detection.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            canvas = np.zeros((250, 300, 3), dtype="uint8")
            if len(faces) > 0:
                face = sorted(faces, reverse=True, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
                (fX, fY, fW, fH) = face
                roi = gray[fY:fY + fH, fX:fX + fW]
                roi = cv2.resize(roi, (48, 48))
                roi = roi.astype("float") / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                preds = emotion_classifier.predict(roi)[0]
                pre = pre + preds
        #cap.release()
        #cv2.destroyAllWindows()

        pre = pre / 30
        emotion_dict = {self.EMOTIONS[i]: value for i, value in enumerate(pre)}
        emotion_sorted = sorted(emotion_dict.items(), key=lambda x: x[1], reverse=True)
        for i in range(0, 6):
            if emotion_sorted[6 - i][1] < 0.16:
                del emotion_sorted[6 - i]
        self.lbl_feeling.setText(self.EMOTIONS_KOR[emotion_sorted[0][0]])

        # 최종
        df = pd.DataFrame()
        for d in range(0, len(emotion_sorted)):
            if d == 0:
                standard = self.emotion_df['new_standard'] > emotion_sorted[d][1]
            else:
                standard = (emotion_sorted[d][1] + 0.1 >= self.emotion_df['new_standard']) & (
                            self.emotion_df['new_standard'] > emotion_sorted[d][1])
            data = (self.emotion_df[(self.emotion_df['feeling'] == emotion_sorted[d][0]) & (standard)])
            df = pd.concat([df, data], ignore_index=True)
        self.emotion_word = df['word'].sample().iloc[0]
        print('감정읽기로 emotion_word: ',self.emotion_word)

    def genre_slot(self):

        if self.classic.isChecked():
            self.genre_word.append('클래식')
            self.genre_word.append('classic')
        if self.bgm.isChecked():
            self.genre_word.append('bgm')
            self.genre_word.append('ost')
        if self.jazz.isChecked():
            self.genre_word.append('재즈 음악')
            self.genre_word.append('jazz')
        if self.k_ballard.isChecked():
            self.genre_word.append('발라드')
        if self.k_pop.isChecked():
            self.genre_word.append('아이돌')
            self.genre_word.append('케이팝')
            self.genre_word.append('K pop')
        if self.k_trot.isChecked():
            self.genre_word.append('트로트')
        if self.k_hiphop.isChecked():
            self.genre_word.append('힙합')
            self.genre_word.append('인디음악')
        try:
            self.genre_word = random.sample(self.genre_word, 1)
            print('음악 장르: ',self.genre_word)
            print('음악 장르: ', type(self.genre_word))
        except:
            self.genre_word = []


    def youtube_play_slot(self):
        self.today_word = self.le_today.text()
        print('today_word: ',self.today_word)
        self.worker.start()

        def youtube_title():
            self.lbl_title.setText(self.worker.title)

        time.sleep(13); youtube_title()

        if self.worker.title == '플레이리스트를 찾는 중입니다':
            time.sleep(4); youtube_title()
        else:
            pass


    def youtube_allstop_slot(self):
        try:
            self.worker.driver.close()
            print('드라이버 닫아버리기')
        except:
            pass

    def youtube_stop_slot(self):
        try:
            self.worker.body.send_keys(Keys.SPACE)
            print('본체에서 타이틀',self.worker.title)
            self.lbl_title.setText(self.worker.title)
        #Keys.Space를 변수로 받아서 stop_flag를 세워서 if stop_flag == True이면 btn.setText('play')설정하기
        except:
            pass


    #창 닫기
    def closeEvent(self, QCloseEvent):
        ans = QMessageBox.question(self, '종료하기', '종료하시겠습니까?', QMessageBox.Yes | QMessageBox.No,
                                   QMessageBox.Yes)
        if ans == QMessageBox.Yes:
            self.worker.driver.close()
            #del self.worker
            QCloseEvent.accept()
        else:
            QCloseEvent.ignore()


app = QApplication(sys.argv)
mainWindow = Exam()
mainWindow.show()

sys.exit(app.exec_())
