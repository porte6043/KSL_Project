
from cProfile import label
from charset_normalizer import detect
import cv2
import sys
import os
import json
import numpy as np
from datetime import datetime

class Frame_Data():
    def __init__(self):
        self.dt = datetime.now()
        self.date = str(self.dt.date()).replace("-","")[2:]

    def jsondata_load(self, path):
        with open(path, 'r', encoding='UTF8') as time:
            self.data_time = json.load(time)

    def extract_frame(self, path, errorpath=None, lenght=None, frame_limit=70, size=0.2, roi_size=0.5, height=1080, width=1920):
        # 프레임 변수
        self.height = height
        self.width = width
        self.image_size = size
        self.roi_size_L = roi_size / 2
        self.roi_size_R = 1 - roi_size / 2

        # 라벨링 변수
        self.label = []
        self.label_count = 0
        self.err_count = 0

        # 에러 제거 및 비디오 이름
        if errorpath != None:
            with open(errorpath, 'r', encoding='UTF8') as errrr:
                self.error_data = json.load(errrr)
            self.error_idxs = list(self.error_data.keys())
            self.error_idxs = [int(i) for i in self.error_idxs]
            self.error_idxs.sort()
        else:
            self.error_idxs = list(self.error_data.keys())
            self.error_idxs.sort()

        self.video_file_names = os.listdir(path)

        for idx , error_idx in enumerate(self.error_idxs):
            error_idx = error_idx - idx
            del self.video_file_names[error_idx*5:(error_idx+1)*5]
            del self.data_time[error_idx*5:(error_idx+1)*5]

        # 길이 제한
        if lenght == None:
            self.data_video = np.zeros( (len(self.video_file_names), frame_limit, int(height*self.image_size), int(width*self.image_size/2)), dtype=np.uint8 )
            lenght = len(self.video_file_names)
        else:
            self.data_video = np.zeros( (lenght, frame_limit, int(height*self.image_size), int(width*self.image_size/2)), dtype=np.uint8 )


        for idx, video_file_name in enumerate(self.video_file_names):
            # 영상 갯수 제한
            if idx == lenght:
                break
            if idx % 100 == 0:
                print("frame count:{}".format(idx))

            # 파라미터 초기화
            self.frame_count = 0 # 프레임 카운트
            self.frame_seq = 0 # 적재 프레임 인덱스

            # 영상 읽어오기
            cap = cv2.VideoCapture(path + video_file_name)
            if not cap.isOpened():
                print("Camera open failed!")
                print(video_file_name)
                continue

            fps = cap.get(cv2.CAP_PROP_FPS)
            start_time = self.data_time[idx][0]
            end_time = self.data_time[idx][1]

            if (end_time-start_time) * fps > frame_limit:
                continue

            while True:
                ret , frame = cap.read()

                if not ret:
                    break

                frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                img = cv2.resize(frame_gray, None, fx=self.image_size, fy=self.image_size, interpolation = cv2.INTER_LINEAR)
                roi = img[: , int(img.shape[1]*self.roi_size_L):int(img.shape[1]*self.roi_size_R)]

                if self.frame_count >= round( fps*start_time ) and self.frame_count <= round( fps*end_time ):
                    self.data_video[idx,self.frame_seq,:,:] = roi
                    self.frame_seq += 1

                if round( fps*end_time ) < self.frame_count:
                    break

                self.frame_count += 1
            
            cap.release() # 사용한 자원 해제

            # 라벨링 작업
            self.Q_inx = idx // 5
            if self.Q_inx in self.error_idxs:
                self.label.append(-1)
                self.label_count += 1
                if self.label_count == 5:
                    self.label_count = 0
                    self.err_count += 1
            else:
                self.label.append(self.Q_inx)
        # 라벨링 작업
        self.label_last = len(self.label) // 5

        for i in range(self.err_count):
            for j in range(5):
                self.label_last = self.detect_err_lable(self.label_last, self.error_idxs)
                self.label.append(self.label_last)
            self.label_last += 1
        
        while -1 in self.label:
            self.label.remove(-1)

    # 에러 검출을 위한 재귀 함수
    def detect_err_lable(self, idx, err_idxs):
        if idx in err_idxs:
            idx = self.detect_err_lable(idx+1, err_idxs)
        return idx

    # 데이터 저장
    def save_data(self, path, name):
        with open(path + '/saved_label_{}_{}_{}x{}'.format(name, len(self.data_video), int(self.height*self.image_size), int(self.width*self.image_size*self.roi_size_L*2)), 'w', encoding ='UTF8') as label:
            json.dump(self.label, label)
        np.save(path + '/saved_image_{}_{}_{}x{}'.format(name, len(self.data_video), int(self.height*self.image_size), int(self.width*self.image_size*self.roi_size_L*2)), self.data_video)

    # 이미지 출력
    def image_show(self, lenght=None, a=True):
        if lenght == None:
            lenght = len(self.data_video)

        print("image_shape:{}".format(self.data_video.shape))
        for i in range(lenght):
            for j in range(60):
                print('{}_{}'.format(i,j))
                cv2.imshow('{}'.format(i), self.data_video[i,j,:,:])
                if cv2.waitKey() == 27:
                    a = False
                    break
            if a == False:
                cv2.destroyAllWindows()
                break
            cv2.destroyAllWindows()
    
    def Error(self, videopath, errorpath, save_err_path=None, name=None, frame_limit=70): # FPS 와 시간차(time_diffrence)를 이용해 최대 프레임 구하기
        with open(errorpath, 'r', encoding='UTF8') as err:
            self.error_data = json.load(err)

        self.time_difference_max = 0 
        self.fps_max = 0
        self.frame_max = 0
        self.frame_max_name = [0]
        self.frame_overflow_name = []
        self.video_file_names = os.listdir(videopath)

        for idx, video_file_name in enumerate(self.video_file_names):
            cap = cv2.VideoCapture(videopath + video_file_name)
            if not cap.isOpened():
                # print("Camera open failed!")
                # print(video_file_name)
                self.error_data[idx//5] = video_file_name
                continue

            fps = cap.get(cv2.CAP_PROP_FPS)
            time_difference = self.data_time[idx][1] - self.data_time[idx][0]
            frame = fps * time_difference

            if fps > self.fps_max:
                self.fps_max = fps

            if time_difference > self.time_difference_max:
                self.time_difference_max = time_difference

            if frame > self.frame_max:
                self.frame_max = frame
                self.frame_max_name[0] = [fps, time_difference, frame, video_file_name]

            if frame > frame_limit:
                self.frame_overflow_name.append(video_file_name)
                self.error_data[idx//5] = video_file_name

            if idx % 1000 == 0:
                print("Errdr_idx:{}".format(idx))

        if save_err_path != None:
            with open(save_err_path+'/saved_err_{}'.format(name), 'w', encoding ='UTF8') as err_data:
                json.dump(self.error_data, err_data, indent=1)

        print('-'*100)
        # print("fps_max:{} , time_diff_max:{} , frame_max:{}".format(self.fps_max, self.time_difference_max, self.frame_max))
        # print("fps_max:{} , time_diff_max:{} , frame_max:{}".format(self.frame_max_name[0][0], self.frame_max_name[0][1], self.frame_max_name[0][2]))


if __name__ == "__main__":
    jsonpath = 'C:/Users/moons/Desktop/Python_code/sign_language/time_220507_10'
    videopath = 'E:/수어 영상/1.Training/[원천]01_real_sen_video/01/'
    FD = Frame_Data()
    FD.jsondata_load(jsonpath)
    FD.extract_frame(videopath)
    FD.save_data()