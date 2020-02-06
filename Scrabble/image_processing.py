import cv2
from math import pi, cos, sin
from tensorflow import keras
from mnist import MNIST
import numpy as np
import imutils


class Image_processor:
    cols = 15
    rows = 15
    nn_index = {
        0: 'a'
    }

    def __init__(self, img_path):
        self.img = cv2.imread(img_path)
        self.crop(self.img, 3, 3)
        self.board = []

    def run(self):
        self.crop(self.img, 3, 3)

    def create_neural_network(self):  # could replace with Hu moment recognition
        data, labels = self.load_data()
        model = keras.Sequential([
            keras.layers.Dense(784, activation="relu"),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dense(27, activation="softmax")
        ])
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        model.fit(data, labels, epochs=1)
        self.nn = model

    def load_data(self):
        # https://github.com/akshaybahadur21/Alphabet-Recognition-EMNIST/blob/master/Alpha-Rec.py
        emnist_data = MNIST(path='data/', return_type='numpy')
        emnist_data.select_emnist('letters')
        imgs, labels = emnist_data.load_training()
        return imgs, labels

    def process_image(self):
        _, thresh = cv2.threshold(self.img, 130, 255, cv2.THRESH_BINARY)
        bw = cv2.bitwise_and(thresh, thresh, mask=self.img)
        # lines = cv2.HoughLinesP(image=bw, rho=1, theta=90, threshold=80, minLineLength=0, maxLineGap=0)
        contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        row = []
        for c in contours:
            rect = cv2.boundingRect(c)
            tile_space = None  # find a way to convert to a pixel array
            index = self.nn.predict(tile_space)
            letter = self.nn_index[index]
            row.append(letter)
            if len(row) > self.cols:
                self.board.append(row)
                row = []

    def template_process(self):
        # multi-scale = https://www.pyimagesearch.com/2015/01/26/multi-scale-template-matching-using-python-opencv/
        # images from: https://www.papertraildesign.com/free-printable-scrabble-letter-tiles-sign/
        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        im_paths = [f"data/Tile_images/Scrabble-tile-{letter}-wood.jpg" for letter in alphabet]
        locations = []
        #im_paths = ["data/a_template.jpg"]
        w = h = 1
        for path, letter in zip(im_paths, alphabet):
            img_rgb = self.img
            img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
            template = cv2.imread(path, 0)
            scale = img_gray.shape[1] / 25
            template = imutils.resize(template, width=int(scale))
            w, h = template.shape[::-1]
            res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
            threshold = 0.5
            loc = np.where(res >= threshold)
            for pt in zip(*loc[::-1]):
                cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)
                locations.append((pt, letter))
        min_y = min(locations, key=lambda x: x[0][1])[0][1]
        min_x = min(locations, key=lambda x: x[0][0])[0][0]
        placed = [(
            (int(round((y-min_y)/h))%self.rows, int(round((x-min_x)/w))%self.cols), letter)
            for (x, y), letter in locations]
        placed = list(set(placed))
        cv2.imwrite('res.png', img_rgb)
        return placed
        print(len(placed))

    def online_image(self):
        # https://stackoverflow.com/questions/48954246/find-sudoku-grid-using-opencv-and-python
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 90, 150, apertureSize=3)
        kernel = np.ones((3,3),np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        kernel = np.ones((5,5),np.uint8)
        edges = cv2.erode(edges, kernel, iterations=1)
        cv2.imwrite('canny.jpg', edges)
        lines = cv2.HoughLinesP(image=edges, rho=1, theta=90, threshold=100, minLineLength=0, maxLineGap=20)
        rho_threshold = 15
        theta_threshold = 0.1

        # how many lines are similar to a given one
        similar_lines = {i: [] for i in range(len(lines))}
        for i in range(len(lines)):
            for j in range(len(lines)):
                if i == j:
                    continue

                rho_i, theta_i = lines[i][0]
                rho_j, theta_j = lines[j][0]
                if abs(rho_i - rho_j) < rho_threshold and abs(theta_i - theta_j) < theta_threshold:
                    similar_lines[i].append(j)

        # ordering the INDECES of the lines by how many are similar to them
        indices = [i for i in range(len(lines))]
        indices.sort(key=lambda x: len(similar_lines[x]))

        # line flags is the base for the filtering
        line_flags = len(lines) * [True]
        for i in range(len(lines) - 1):
            if not line_flags[indices[
                i]]:  # if we already disregarded the ith element in the ordered list then we don't care (we will not delete anything based on it and we will never reconsider using this line again)
                continue

            for j in range(i + 1, len(lines)):  # we are only considering those elements that had less similar line
                if not line_flags[indices[j]]:  # and only if we have not disregarded them already
                    continue

                rho_i, theta_i = lines[indices[i]][0]
                rho_j, theta_j = lines[indices[j]][0]
                if abs(rho_i - rho_j) < rho_threshold and abs(theta_i - theta_j) < theta_threshold:
                    line_flags[
                        indices[j]] = False  # if it is similar and have not been disregarded yet then drop it now
        filtered_lines = []
        for i in range(len(lines)):  # filtering
            if line_flags[i]:
                filtered_lines.append(lines[i])

        print('Number of filtered lines:', len(filtered_lines))
        for line in filtered_lines:
            rho, theta = line[0]
            a = cos(theta)
            b = sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            cv2.line(self.img, (x1, y1), (x2, y2), (0, 0, 255), 2)

        cv2.imwrite('hough.jpg', self.img)

    def canny_test(self):
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 90, 150, apertureSize=3)
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        kernel = np.ones((5, 5), np.uint8)
        edges = cv2.erode(edges, kernel, iterations=1)
        blurred = cv2.GaussianBlur(edges, (11, 11), 0)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [cv2.convexHull(c) for c in contours]
        largest = max(contours, key=lambda x: cv2.contourArea(x))
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(self.img, (x, y), (x+w, y+h), (0, 0, 255), 5)
        cv2.drawContours(self.img, contours, -1, (255, 0, 9), 1)  # draws the shapes that it finds
        cv2.imwrite('canny.jpg', edges)

    def mser(self):
        img = self.img
        img = abs(255 - img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ## Get mser, and set parameters
        mser = cv2.MSER_create()
        mser.setMinArea(100)
        mser.setMaxArea(1000)

        ## Do mser detection, get the coodinates and bboxes
        coordinates, bboxes = mser.detectRegions(gray)

        ## Filter the coordinates
        vis = img.copy()
        coords = []
        for coord in coordinates:
            bbox = cv2.boundingRect(coord)
            x, y, w, h = bbox
            # cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 3)
            if w < 10 or h < 10 or w / h > 5 or h / w > 5:
                continue
            coords.append(coord)

        ## colors
        colors = [[43, 43, 200], [43, 75, 200], [43, 106, 200], [43, 137, 200], [43, 169, 200], [43, 200, 195],
                  [43, 200, 163], [43, 200, 132], [43, 200, 101], [43, 200, 69], [54, 200, 43], [85, 200, 43],
                  [116, 200, 43], [148, 200, 43], [179, 200, 43], [200, 184, 43], [200, 153, 43], [200, 122, 43],
                  [200, 90, 43], [200, 59, 43], [200, 43, 64], [200, 43, 95], [200, 43, 127], [200, 43, 158],
                  [200, 43, 190], [174, 43, 200], [142, 43, 200], [111, 43, 200], [80, 43, 200], [43, 43, 200]]

        ## Fill with random colors
        np.random.seed(0)
        canvas1 = img.copy()
        canvas2 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        canvas3 = np.zeros_like(img)

        for cnt in coords:
            xx = cnt[:, 0]
            yy = cnt[:, 1]
            color = colors[np.random.choice(len(colors))]
            canvas1[yy, xx] = color
            canvas2[yy, xx] = color
            canvas3[yy, xx] = color

        ## Save
        cv2.imwrite("result1.png", canvas1)
        cv2.imwrite("result2.png", canvas2)
        cv2.imwrite("result3.png", canvas3)

    def brisk(self):
        # https://www.youtube.com/watch?v=uwN0JAY548M
        keypoints = cv2.BRISK_create()
        # detectAndcompute
        # knn match
        # findHomography
        # inverse perspective transform
        # grey blur
        # adaptive threshold
        # dilate

    def crop(self, img, rows, cols):
        h, w = img.shape[:2]
        x_step = w/cols
        y_step = h/rows
        imgs = []
        for r in range(rows):
            for c in range(cols):
                x1 = int(c*x_step)
                y1 = int(r*y_step)
                x2 = int(x1 + x_step)
                y2 = int(y1 + y_step)
                section = img[y1: y2, x1: x2]
                imgs.append(section)
        cv2.imshow("cropped.jpg", imgs[0])