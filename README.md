# youtube
play youtube music

유튜브에서 백그라운드로 음악을 재생하는 프로그램을 만들고자 한다.

먼저 여기에서 참고한 git은 다음과 같다.

얼굴인식 기능 : https://github.com/codeingschool/Facial-Recognition

얼굴인식을 바탕으로한 감정인식 기능 : https://github.com/omar178/Emotion-recognition

한국어 감정 사전 출처 : 논문 참조 - 손선주 외 4명, 한국어 감정표현단어의 추출과 범주화, 2012 청주대학교 사회복지학과   


# 구현하고자 한 프로그램

1. 로그인 창 구현 : 일반 로그인, 얼굴 인식으로 로그인

2. 메인 UI 검색 키워드 설정 : 음악 종류로 키워드 검색, 오늘의 한마디로 키워드 검색, 감정을 기반으로 키워드 검색

3. 메인 UI 음악 재생 : 음악 재생 프로그램 구현, 현재 음악 재생 목록을 보여줌


# 개선사항

1. thread를 이용한 방식이 아닌 다른 방식으로 접근해보기

2. 다음곡으로 넘어가거나 볼륨을 조절하는 등 다양한 UI기능을 넣고자 함.

3. 단어 입력값에 따른 아웃풋(랜덤) 값 설정을 좀 더 디테일 하게 하고 싶음.


# 먼저 얼굴 인식을 할 로그인 데이터 세팅



필수 파일:

chromedriver : 최신거로 다운받아도 된다.

login_try : 로그인 하는 화면의 UI

music_base : 음악재생프로그램의 UI

music_final_2 : 파일을 실행하면 구현

music_ppt : 사용설명서 ? 



<login_datasets> 폴더 사용

customer.xlsx : 각 고객의 정보다 해당 아이디와 비밀번호가 입력되어있다.

emotion_model.hdf5 : 감정인식 모델

haarcascade_frontalface_default : 얼굴인식하는 모델

Part1 : 파일을 실행하면 웹캠에 보이는 얼굴 사진을 200장 찍고, faces 폴더 안에 각 user 이름에 맞게 폴더를 생성하고 사진을 저장한다.

Part2 : 모델을 학습시킨다. 경로를 잘 봐야한다. yml 모델로 만든다.

