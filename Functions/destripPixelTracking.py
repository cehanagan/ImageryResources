import sys
import numpy as np
from osgeo import gdal
import math
import cv2
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QPushButton,
                             QLabel, QLineEdit, QFileDialog)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

class GeoTiffPlotter(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.points = []
        self.min_value = None
        self.max_value = None
        self.final_image = None  # Initialize final_image

    def initUI(self):
        self.setWindowTitle('GeoTIFF Plotter')

        self.layout = QVBoxLayout()

        # Create a layout for buttons
        self.button_layout = QVBoxLayout()

        self.load_button = QPushButton('Load GeoTIFF')
        self.load_button.clicked.connect(self.load_geotiff)
        self.button_layout.addWidget(self.load_button)

        self.min_value_input = QLineEdit(self)
        self.min_value_input.setPlaceholderText('Enter Min Color Value')
        self.button_layout.addWidget(self.min_value_input)

        self.max_value_input = QLineEdit(self)
        self.max_value_input.setPlaceholderText('Enter Max Color Value')
        self.button_layout.addWidget(self.max_value_input)

        self.replot_button = QPushButton('Replot')
        self.replot_button.clicked.connect(self.replot)
        self.button_layout.addWidget(self.replot_button)

        self.destripe_button = QPushButton('Destripe Image')
        self.destripe_button.clicked.connect(self.destripe_image)
        self.button_layout.addWidget(self.destripe_button)


        self.save_image_button = QPushButton('Save Image')  # New Save Image button
        self.save_image_button.clicked.connect(self.save_image)
        self.button_layout.addWidget(self.save_image_button)

        self.delete_button = QPushButton('Delete Last Point')
        self.delete_button.clicked.connect(self.delete_last_point)
        self.button_layout.addWidget(self.delete_button)

        self.exit_button = QPushButton('Exit')
        self.exit_button.clicked.connect(self.close)
        self.button_layout.addWidget(self.exit_button)

        self.layout.addLayout(self.button_layout)

        self.canvas = FigureCanvas(plt.figure())
        self.layout.addWidget(self.canvas)

        self.setLayout(self.layout)

        self.canvas.mpl_connect('button_press_event', self.on_click)

    def load_geotiff(self):
        file_path, _ = QFileDialog.getOpenFileName(self, 'Open GeoTIFF File', '', 'GeoTIFF Files (*.tif *.tiff)')
        if file_path:
            self.dataset = gdal.Open(file_path)
            self.data = self.dataset.GetRasterBand(1).ReadAsArray()
            #self.transform = self.dataset.transform
            self.replot()

    def replot(self):
        min_val = float(self.min_value_input.text()) if self.min_value_input.text() else None
        max_val = float(self.max_value_input.text()) if self.max_value_input.text() else None

        plt.clf()
        plt.imshow(self.data, vmin=min_val, vmax=max_val)
        
        if self.points:
            x, y = zip(*self.points)
            plt.plot(x, y, 'ro-')  # Red points with lines connecting them

        self.canvas.draw()

    def on_click(self, event):
        if event.inaxes is not None:
            x = event.xdata
            y = event.ydata
            self.points.append((x, y))
            self.replot()

   
    def delete_last_point(self):
        if self.points:
            self.points.pop()
            self.replot()

    def destripe_image(self):
        nodata = np.nan
        if len(self.points) < 2:
            print("Need at least 2 points to define a line.")
            return

        # Calculate the azimuth of the line defined by the points
        p1, p2 = self.points[0], self.points[1]
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        azimuth = np.arctan2(dx, dy) * (180 / np.pi)  # Convert to degrees
        angleInDegrees = azimuth - 90  # Rotate to make the line horizontal

        # Rotate the image so the stripes are vertical (stripes are columns)
        h, w = self.data.shape[:2]
        img_c = (w / 2, h / 2)
        rot = cv2.getRotationMatrix2D(img_c, -angleInDegrees, 1)
        rad = math.radians(angleInDegrees)
        sin = math.sin(rad)
        cos = math.cos(rad)
        b_w = int((h * abs(sin)) + (w * abs(cos)))
        b_h = int((h * abs(cos)) + (w * abs(sin)))
        # Create some padding so the edges of the image aren't lost
        rot[0, 2] += ((b_w / 2) - img_c[0])
        rot[1, 2] += ((b_h / 2) - img_c[1])
        rotImg = cv2.warpAffine(self.data, rot, (b_w, b_h), flags=cv2.INTER_LINEAR)
        

        # Calculate the median of each column, ignoring NaN values
        means = np.nanmedian(rotImg, axis=1)

        # Subtract the mean from each column
        destriped_image = rotImg - means[:, np.newaxis]

        # Rotate back to original orientation, creating additional padding
        h, w = destriped_image.shape[:2]
        img_c = (w / 2, h / 2)
        rot = cv2.getRotationMatrix2D(img_c, angleInDegrees, 1)
        rad = math.radians(angleInDegrees)
        sin = math.sin(rad)
        cos = math.cos(rad)
        b_w = int((h * abs(sin)) + (w * abs(cos)))
        b_h = int((h * abs(cos)) + (w * abs(sin)))

        rot[0, 2] += ((b_w / 2) - img_c[0])
        rot[1, 2] += ((b_h / 2) - img_c[1])
        destripedImg = cv2.warpAffine(destriped_image, rot, (b_w, b_h), flags=cv2.INTER_LINEAR)
         
        # Crop rotated image to the original size (remove the added padding)
        center_x, center_y = np.array(destripedImg.shape) // 2
        orig_x, orig_y = np.array(self.data.shape) // 2
        cropped_data = destripedImg[
            center_x - orig_x : center_x + orig_x,
            center_y - orig_y : center_y + orig_y
        ]


        # Update the data with the destriped image and replot
        self.final_image = cropped_data
        self.data = cropped_data
        self.replot()

    def save_image(self):
        if self.final_image is not None:
            file_path, _ = QFileDialog.getSaveFileName(self, 'Save GeoTIFF', '', 'GeoTIFF Files (*.tif *.tiff)')
            if file_path:
                rows, cols = self.data.shape
                driver = gdal.GetDriverByName('GTIFF')
                out_data = driver.Create(file_path, cols, rows, 1, gdal.GDT_Float32)
                # Set the geotransform and projection
                out_data.SetGeoTransform(self.dataset.GetGeoTransform())
                out_data.SetProjection(self.dataset.GetProjection())
                out_data.GetRasterBand(1).SetNoDataValue(0)

                # Write the data to the band
                out_data.GetRasterBand(1).WriteArray(self.final_image)

                # Close the file
                out_data = None
    
               
                print(f"Image saved to {file_path}")
        else:
            print("No destriped image to save.")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = GeoTiffPlotter()
    ex.show()
    sys.exit(app.exec_())