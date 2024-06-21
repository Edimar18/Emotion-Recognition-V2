from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5 import QtTest
from PyQt5 import uic
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import pickle as pkl


class WINDOW(QMainWindow):
    def __init__(self):
        super(WINDOW, self).__init__()
        uic.loadUi("layout.ui", self)
        self.show()
        
        ## VARIABLES
        self.analysesOn = True
        self.predict = True
        self.faceMesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mpDraw = mp.solutions.drawing_utils
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.figure.set_facecolor('#B3C8CF')
        self.model = pkl.load(open('clf_emotionRecogV3.pkl', 'rb'))
        
        ## BUTTON CLICK HANDLER
        self.pushButton.clicked.connect(self.toggleAnalysesOn)
        self.pushButton_2.clicked.connect(self.toggleAnalysesOff)  
        
        self.cap = cv2.VideoCapture('sign2.mp4')
        while True:
            ret, frame = self.cap.read()
            if ret:
                self.image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                #self.image.resize((591, 751))
                height, width, channel = self.image.shape
                step = channel * width
                qImg = QImage(self.image.data, width, height, step, QImage.Format_RGB888)
                self.frame = qImg.scaled(591, 751, Qt.KeepAspectRatio)
                self.label_5.setPixmap(QPixmap.fromImage(self.frame))
                
                if self.analysesOn:
                    self.mesh = self.faceMesh.process(self.image)
                    if self.mesh.multi_face_landmarks:
                        LANDMARKS = []
                        
                        
                        for faceLms in self.mesh.multi_face_landmarks:
                            x_ = []
                            y_ = []
                            z_ = []
                            lips_right = [164, 0 , 11, 12, 13, 14, 15, 16, 17, 18, 393, 267, 302, 268, 312, 317, 316, 315, 314, 113, 391, 269, 303, 271, 311, 402, 403, 404, 405, 406, 322, 270, 304, 272, 310, 319, 320, 321, 335, 410, 409, 408, 407, 415, 324, 325, 307, 375, 273, 308, 287, 436, 432]
                            lips_left = [83, 84, 85, 86, 87, 82, 38, 72, 37, 167, 165, 39, 73, 41, 81, 178, 179, 180, 181, 182, 106, 91, 90, 89, 88, 80, 42, 74, 40, 185, 184, 183, 191, 95, 96, 77, 146, 76, 62, 61, 57]
                            eyes_left = [244, 190, 243, 233, 173, 155, 112, 232, 56, 157, 154, 26, 28, 158, 153, 22, 27, 159, 145, 23, 24, 144, 160, 29, 30, 161, 163,110, 25, 7, 246, 247, 33, 25, 130]
                            facePoints = lips_right + lips_left  + eyes_left
                            for i in range(len(faceLms.landmark)):
                                x_.append(faceLms.landmark[i].x)
                                y_.append(faceLms.landmark[i].y)
                                z_.append(faceLms.landmark[i].z)
                            
                            for i in range(len(x_)):
                                LANDMARKS.append(x_[i] - min(x_))
                                LANDMARKS.append((height - y_[i]) - min(y_))
                                LANDMARKS.append(z_[i]  - min(z_))
                            
                        self.ax.clear()
                        self.ax.scatter(LANDMARKS[0::3],LANDMARKS[1::3], cmap='viridis', c=LANDMARKS[2::3], s=10)
                        self.canvas.draw()
                        width, height = int(self.figure.get_size_inches()[0] * self.figure.get_dpi()), int(self.figure.get_size_inches()[1] * self.figure.get_dpi())
                        imageData = self.canvas.buffer_rgba().tobytes()
                        image = QImage(imageData, int(width), int(height), QImage.Format_ARGB32)
                        scaled_img = image.scaled(591, 751, Qt.KeepAspectRatio)
                        self.label_6.setPixmap(QPixmap.fromImage(scaled_img))
                        
                        if self.predict:
                            prob = self.model.predict_proba([LANDMARKS])
                            self.progressBar.setValue(int(prob[0][0]*100))
                            self.progressBar_2.setValue(int(prob[0][1]*100))
                            self.progressBar_3.setValue(int(prob[0][2]*100))
                            self.progressBar_4.setValue(int(prob[0][3]*100))
                            self.progressBar_5.setValue(int(prob[0][4]*100))
                            self.progressBar_6.setValue(int(prob[0][5]*100))
                            self.progressBar_7.setValue(int(prob[0][6]*100))

                    
                        
                if self.analysesOn:
                    QtTest.QTest.qWait(1)
                else:
                    QtTest.QTest.qWait(10)
    def toggleAnalysesOn(self):
        self.analysesOn = True
        self.pushButton.setStyleSheet("background-color: rgb(255, 44, 118);border-radius: 2px;font: italic 8pt '8514oem';")
        self.pushButton_2.setStyleSheet("background-color: rgb(184, 184, 184);border-radius: 2px;font: italic 8pt '8514oem';")
    def toggleAnalysesOff(self):
        self.analysesOn = False
        self.pushButton_2.setStyleSheet("background-color: rgb(255, 44, 118);border-radius: 2px;font: italic 8pt '8514oem';")
        self.pushButton.setStyleSheet("background-color: rgb(184, 184, 184);border-radius: 2px;font: italic 8pt '8514oem';")
        self.label_6.setText(' ')
        
                
        
if __name__ == "__main__":
    app = QApplication([])
    window = WINDOW()
    app.exec_()