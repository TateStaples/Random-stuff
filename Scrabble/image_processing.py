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
        gray = cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY)
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

    def template_identify(self, img):
        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        im_paths = [f"data/Tile_images/Scrabble-tile-{letter}-wood.jpg" for letter in alphabet]
        max_val = 0
        best_letter = ""
        for path, letter in zip(im_paths, alphabet):
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            template = cv2.imread(path, 0)
            scale = img_gray.shape[1] / 25
            template = imutils.resize(template, width=int(scale))
            w, h = template.shape[::-1]
            res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
            (_, maxVal, _, maxLoc) = cv2.minMaxLoc(res)
            if maxVal > max_val:
                max_val = maxVal
                best_letter = letter
        return best_letter
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

    def generate_corners(self, row, col):
        tl = np.array([[1, 1],
                        [1, 0]])
        br = np.array([[0, 1],
                       [1, 1]])
        tr = np.array([[1, 1],
                       [0, 1]])
        bl = np.array([[1, 0],
                       [1, 1]])
        tl = np.kron(tl, np.ones((row, col)))
        br = np.kron(br, np.ones((row, col)))
        tr = np.kron(tr, np.ones((row, col)))
        bl = np.kron(bl, np.ones((row, col)))
        #print(tl.shape)
        tl *= 255
        br *= 255
        bl *= 255
        tr *= 255
        tl = cv2.cvtColor(tl.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        br = cv2.cvtColor(br.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        bl = cv2.cvtColor(bl.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        tr = cv2.cvtColor(tr.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        return tl, tr, bl, br

    def grid(self):
        (x1, y1), (x2, y2) = self.get_corners()
        img = self.img[y1:y2, x1:x2]
        self.draw_grid(img, self.rows, self.cols)
        imgs = self.crop(img, self.rows, self.cols)
        cv2.imwrite("section.jpg", imgs[63])
        imgs[63] = cv2.cvtColor(imgs[63], cv2.COLOR_RGB2GRAY)
        self.create_neural_network()
        print(self.nn.predict([cv2.resize(imgs[63], (28, 28))]))
        print(self.template_identify(imgs[63]))

    def get_corners(self):
        size = 27
        tl, tr, bl, br = self.generate_corners(size, size)
        size *= 2
        #self.multi_scale_template(self.img, tl)
        result = cv2.matchTemplate(self.img, tl, cv2.TM_CCOEFF)
        (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
        x1, y1 = maxLoc
        rect(self.img, x1, y1, size, size, (0, 255, 0))
        result = cv2.matchTemplate(self.img, tr, cv2.TM_CCOEFF)
        (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
        x2, y2 = maxLoc
        rect(self.img, x2, y2, size, size, (0, 255, 0))
        result = cv2.matchTemplate(self.img, bl, cv2.TM_CCOEFF)
        (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
        x3, y3 = maxLoc
        rect(self.img, x3, y3, size, size, (0, 255, 0))
        result = cv2.matchTemplate(self.img, br, cv2.TM_CCOEFF)
        (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
        x4, y4 = maxLoc
        rect(self.img, x4, y4, size, size, (0, 255, 0))
        x1, y1 = min([x1+size, x3+size]), min([y1+size, y2+size])
        x2, y2 = max([x2, x4]), max([y3, y4])
        cv2.rectangle(self.img, (x1, y1), (x2, y2), (255, 0, 0), 3)
        cv2.imwrite("corners.jpg", self.img)
        return (x1, y1), (x2, y2)


    def multi_scale_template(self, img, template):
        template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        template = cv2.Canny(template, 50, 200)
        (tH, tW) = template.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        found = None
        # loop over the scales of the image
        for scale in np.linspace(0.2, 1.0, 20)[::-1]:
            # resize the image according to the scale, and keep track
            # of the ratio of the resizing
            resized = imutils.resize(gray, width=int(gray.shape[1] * scale))
            r = gray.shape[1] / float(resized.shape[1])
            # if the resized image is smaller than the template, then break
            # from the loop
            if resized.shape[0] < tH or resized.shape[1] < tW:
                break
            # detect edges in the resized, grayscale image and apply template
            # matching to find the template in the image
            edged = cv2.Canny(resized, 50, 200)
            result = cv2.matchTemplate(edged, template, cv2.TM_CCOEFF)
            (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
            # if we have found a new maximum correlation value, then update
            # the bookkeeping variable
            if found is None or maxVal > found[0]:
                found = (maxVal, maxLoc, r)
        # unpack the bookkeeping variable and compute the (x, y) coordinates
        # of the bounding box based on the resized ratio
        (_, maxLoc, r) = found
        (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
        (endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))
        # draw a bounding box around the detected result and display the image
        cv2.rectangle(img, (startX, startY), (endX, endY), (0, 0, 255), 5)
        print(startX, endX)
        cv2.imwrite("Image.jpg", img)

    @staticmethod
    def draw_grid(img, rows, cols):
        h, w = img.shape[:2]
        x_step = w/cols
        y_step = h/rows
        for c in range(1, cols):
            x = int(c * x_step)
            cv2.line(img, (x, 0), (x, h-1), (0, 255, 0), 3)
        for r in range(1, rows):
            y = int(r*y_step)
            cv2.line(img, (0, y), (w-1, y), (0, 255, 0), 3)
        cv2.imwrite("grid.jpg", img)

    @staticmethod
    def crop(img, rows, cols):
        h, w = img.shape[:2]
        x_step = w / cols
        y_step = h / rows
        imgs = []
        for r in range(rows):
            for c in range(cols):
                x1 = int(c * x_step)
                y1 = int(r * y_step)
                x2 = int(x1 + x_step)
                y2 = int(y1 + y_step)
                section = img[y1: y2, x1: x2]
                imgs.append(section)
        return imgs

    def perspective_warp(self):
        # Read the images to be aligned
        im1 = cv2.imread("data/empty_scrabble.jpg")
        im2 = cv2.imread("data/angled_scrabble.jpg")

        # Convert images to grayscale
        im1_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
        im2_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

        # Find size of image1
        sz = im1.shape

        # Define the motion model
        warp_mode = cv2.MOTION_TRANSLATION

        # Define 2x3 or 3x3 matrices and initialize the matrix to identity
        if warp_mode == cv2.MOTION_HOMOGRAPHY:
            warp_matrix = np.eye(3, 3, dtype=np.float32)
        else:
            warp_matrix = np.eye(2, 3, dtype=np.float32)

        # Specify the number of iterations.
        number_of_iterations = 5000

        # Specify the threshold of the increment
        # in the correlation coefficient between two iterations
        termination_eps = 1e-10

        # Define termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)

        # Run the ECC algorithm. The results are stored in warp_matrix.
        (cc, warp_matrix) = cv2.findTransformECC(im1_gray, im2_gray, warp_matrix, warp_mode, criteria, 1, 1)

        if warp_mode == cv2.MOTION_HOMOGRAPHY:
            # Use warpPerspective for Homography
            im2_aligned = cv2.warpPerspective(im2, warp_matrix, (sz[1], sz[0]),
                                              flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        else:
            # Use warpAffine for Translation, Euclidean and Affine
            im2_aligned = cv2.warpAffine(im2, warp_matrix, (sz[1], sz[0]),
                                         flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

        # Show final results
        cv2.imwrite("Image_1.jpg", im1)
        cv2.imwrite("Image_2.jpg", im2)
        cv2.imwrite("Aligned_Image_2.jpg", im2_aligned)

def rect(i, x, y, w, h, c):
    cv2.rectangle(i, (x, y), (x+w, y+h), c)