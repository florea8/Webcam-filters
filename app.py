from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QPushButton, QSlider, QWidget
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QPixmap, QImage
import cv2
import numpy as np

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.updateFrame)

        self.setupUi()

    def setupUi(self):
        self.setWindowTitle("Procesare webcam")
        self.setGeometry(100, 100, 1200, 900)

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        self.label_3 = QLabel(self)
        layout.addWidget(self.label_3)

        self.horizontalSlider = QSlider(Qt.Horizontal)
        layout.addWidget(self.horizontalSlider)
        self.horizontalSlider.valueChanged.connect(self.updateFrame)

        self.pushButtonBlur = QPushButton("Blur", self)
        self.pushButtonBlur.clicked.connect(lambda: self.on_pushButton_clicked("blur"))
        layout.addWidget(self.pushButtonBlur)

        self.pushButton_GBlur = QPushButton("Gaussian Blur", self)
        self.pushButton_GBlur.clicked.connect(lambda: self.on_pushButton_clicked("gaussian_blur"))
        layout.addWidget(self.pushButton_GBlur)

        self.pushButton_MedBlur = QPushButton("Median Blur", self)
        self.pushButton_MedBlur.clicked.connect(lambda: self.on_pushButton_clicked("median_blur"))
        layout.addWidget(self.pushButton_MedBlur)

        self.SobelVertical = QPushButton("Sobel Vertical", self)
        self.SobelVertical.clicked.connect(lambda: self.on_pushButton_clicked("sobel_vertical"))
        layout.addWidget(self.SobelVertical)

        self.SobelOrizontal = QPushButton("Sobel Horizontal", self)
        self.SobelOrizontal.clicked.connect(lambda: self.on_pushButton_clicked("sobel_horizontal"))
        layout.addWidget(self.SobelOrizontal)

        self.FiltruBilateral = QPushButton("Bilateral Filter", self)
        self.FiltruBilateral.clicked.connect(lambda: self.on_pushButton_clicked("bilateral_filter"))
        layout.addWidget(self.FiltruBilateral)

        self.FilterType = "blur"
        self.kernelSize = 1
        self.videoCapture = cv2.VideoCapture(0)
        self.videoCapture.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
        self.videoCapture.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

        self.timer.start(30)

    def on_pushButton_clicked(self, filter_type):
        self.FilterType = filter_type
        self.updateFrame()  

    def updateFrame(self):
        self.kernelSize = max(1, self.horizontalSlider.value())
        self.kernelSize = self.kernelSize + 1 if self.kernelSize % 2 == 0 else self.kernelSize

        ret, frame = self.videoCapture.read()
        if ret:
            filtered_frame = self.apply_filter(frame)
            self.display_image(filtered_frame, self.label_3)

    def apply_filter(self, frame):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        if self.FilterType == "blur":
            filtered_frame = cv2.blur(frame, (self.kernelSize, self.kernelSize))
        elif self.FilterType == "gaussian_blur":
            filtered_frame = cv2.GaussianBlur(frame, (self.kernelSize, self.kernelSize), 0, 0)
        elif self.FilterType == "median_blur":
            filtered_frame = cv2.medianBlur(frame, self.kernelSize)
        elif self.FilterType in ["sobel_vertical", "sobel_horizontal"]:
            kernel_size = max(1, self.horizontalSlider.value())
            kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size

            if self.FilterType == "sobel_vertical":
                gradient = cv2.Sobel(gray_frame, cv2.CV_64F, 0, 1, ksize=kernel_size)
            else: 
                gradient = cv2.Sobel(gray_frame, cv2.CV_64F, 1, 0, ksize=kernel_size)

            abs_gradient = cv2.convertScaleAbs(gradient)
            filtered_frame = cv2.cvtColor(abs_gradient, cv2.COLOR_GRAY2BGR)
        elif self.FilterType == "bilateral_filter":
            filtered_frame = cv2.bilateralFilter(frame, 10, self.horizontalSlider.value(), self.horizontalSlider.value())
        else:
            filtered_frame = frame.copy()

        return filtered_frame


    def display_image(self, image, label):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (label.width(), label.height()))

        q_img = QImage(image.data, image.shape[1], image.shape[0], image.strides[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        label.setPixmap(pixmap)


if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()