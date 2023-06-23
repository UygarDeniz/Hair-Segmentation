import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFileDialog, QFrame, QColorDialog, QSlider
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

import change_hair_color

from main import predict_mask,model
import numpy as np
import cv2
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("App")
        self.setGeometry(100, 100, 1920, 1080)

        self.initUI()
    def initUI(self):
        main_layout = QHBoxLayout()
        self.setLayout(main_layout)

        main_Widget = QWidget()
        main_Widget.setStyleSheet("background-color : #000000")
        main_layout.addWidget(main_Widget)

        left_column_layout = QVBoxLayout()
        main_layout.addLayout(left_column_layout, stretch=3)

        top_row_layout = QHBoxLayout()
        left_column_layout.addLayout(top_row_layout)

        color_button = QPushButton("Select Color")
        color_button.clicked.connect(self.select_color)
        color_button.setStyleSheet("background-color: #FFD100")
        top_row_layout.addWidget(color_button)

        bottom_row_layout = QHBoxLayout()
        left_column_layout.addLayout(bottom_row_layout)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 50)
        self.slider.setSliderPosition(20)
        self.slider.valueChanged.connect(self.slider_value_changed)
        bottom_row_layout.addWidget(self.slider)

        right_column_layout = QVBoxLayout()
        main_layout.addLayout(right_column_layout, stretch=7)

        photo_frame = QFrame()
        photo_frame.setFrameShape(QFrame.Box)
        photo_frame.setLineWidth(1)
        right_column_layout.addWidget(photo_frame, alignment=Qt.AlignCenter)

        self.photo_label = QLabel()
        self.photo_label.setFixedSize(1024, 1024)
        self.photo_label.setAlignment(Qt.AlignCenter)
        self.photo_label.setStyleSheet("border: 1px dashed black;")
        photo_frame_layout = QVBoxLayout(photo_frame)
        photo_frame_layout.addWidget(self.photo_label)

        buttons_layout = QHBoxLayout()

        button1 = QPushButton("Select Photo")
        button1.clicked.connect(self.select_photo)
        button1.setFixedSize(80, 30)
        button1.setStyleSheet("background-color : #008BFF")
        buttons_layout.addWidget(button1)

        button2 = QPushButton("Save")
        button2.clicked.connect(self.save_photo)
        button2.setFixedSize(80, 30)
        button2.setStyleSheet("background-color: #008BFF")
        buttons_layout.addWidget(button2)

        button3 = QPushButton("Reset")
        button3.clicked.connect(self.reset_photo)
        button3.setFixedSize(80, 30)
        button3.setStyleSheet("background-color: #008BFF")
        buttons_layout.addWidget(button3)

        right_column_layout.addLayout(buttons_layout)

        padding_widget = QWidget()
        padding_widget.setFixedHeight(30)
        right_column_layout.addWidget(padding_widget)
    def select_photo(self):
        file_dialog = QFileDialog()
        file_dialog.setNameFilter("Resim Dosyaları (*.png *.jpg *.jpeg)")
        if file_dialog.exec_():
            selected_file = file_dialog.selectedFiles()[0]
            pixmap = QPixmap(selected_file)
            self.photo_label.setPixmap(pixmap)
            self.pixmap_original = pixmap.copy()
    def select_color(self):
        color_dialog = QColorDialog()
        color = color_dialog.getColor()
        self.color_bgr = (color.blue(), color.green(), color.red())
        print(self.color_bgr)
        if color.isValid() and self.photo_label.pixmap() is not None:
            self.photo_label.setPixmap(self.pixmap_original)
            image = self.pixmapToImage()
            self.mask = predict_mask(image, model)
            changed_hair = change_hair_color.change_hair_color(image,
                                                               self.mask,
                                                               self.color_bgr,
                                                               alpha = self.slider.value() / 100)

            pixmap_result = self.ImageToPixmap(image)
            self.photo_label.setPixmap(pixmap_result)
    def save_photo(self):
        pixmap = self.photo_label.pixmap()
        if pixmap is not None:
            try:
                image = pixmap.toImage().convertToFormat(QImage.Format_RGB888)
                file_dialog = QFileDialog()
                file_dialog.setNameFilter("Resim Dosyaları (*.png *.jpg *.jpeg)")
                if file_dialog.exec_():
                    selected_file = file_dialog.selectedFiles()[0]
                    image.save(selected_file)
            except Exception as e:
                print(str(e))
    def reset_photo(self):
        self.slider.setValue(20)
        self.photo_label.setPixmap(self.pixmap_original)
    def slider_value_changed(self):
        if self.photo_label.pixmap() is not None:
            self.photo_label.setPixmap(self.pixmap_original)
            value = self.slider.value() / 100
            image = self.pixmapToImage()
            changed_hair = change_hair_color.change_hair_color(image, self.mask, self.color_bgr, value)
            result = self.ImageToPixmap(changed_hair)
            self.photo_label.setPixmap(result)
    def pixmapToImage(self):
        pixmap = self.photo_label.pixmap()
        image = pixmap.toImage().convertToFormat(QImage.Format_RGB888)

        width = image.width()
        height = image.height()

        buffer = image.constBits()
        buffer.setsize(image.byteCount())
        result = np.array(buffer).reshape(height, width, 3)
        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        return result
    def ImageToPixmap(self, image):
        changed_hair = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        changed_hair_QImage = QImage(changed_hair.data,
                                     changed_hair.shape[1],
                                     changed_hair.shape[0],
                                     QImage.Format_RGB888)
        pixmap_result = QPixmap.fromImage(changed_hair_QImage)
        return pixmap_result

app = QApplication(sys.argv)
window = MainWindow()
window.show()
sys.exit(app.exec_())
