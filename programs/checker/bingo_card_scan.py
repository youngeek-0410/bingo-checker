"Usage: python -m checker.bingo_card_scan"

import cv2
import numpy as np

from PIL import Image
import pyocr
import pyocr.builders

from .checker import Checker
import time
import sys
import tensorflow as tf


class BingoCardScanner:
    LINE_WIDTH = 3
    RED = (0, 0, 255)
    GREEN = (0, 255, 0)
    BLUE = (255, 0, 0)

    # BINGOカードの外枠の白色
    WHITE_MIN_RGB = (130, 130, 130)
    WHITE_MAX_RGB = (255, 255, 255)

    # 使用するMLモデルのパス (日付 / モデル名)
    LOAD_MODEL_FOLDER_NAME = "20220117-214848"
    LOAD_MODEL_FILE_NAME = "weight-28-0.301-0.942-0.123-0.987.h5"

    def __init__(
            self,
            is_use_servo=False,
            ocr_times=5,
            servo_sleep_time=0.3,
            is_use_ocr=False):
        self.ocr_times = ocr_times
        self.servo_sleep_time = servo_sleep_time

        self.is_use_servo = is_use_servo
        if is_use_servo:
            from .servo import Servo
            self.servo = Servo()

        # OCR
        self.is_use_ocr = is_use_ocr
        if is_use_ocr:
            tools = pyocr.get_available_tools()
            if len(tools) == 0:
                print("No OCR tool found")
                return
            self.ocr_tool = tools[0]
            self.ocr_builder = pyocr.builders.TextBuilder(tesseract_layout=6)
            self.ocr_builder.tesseract_configs.append("digits")
        else:
            self.model = tf.keras.models.load_model(
                f"./ML/models/{self.LOAD_MODEL_FOLDER_NAME}/{self.LOAD_MODEL_FILE_NAME}",
                compile=False)
            with open(f"./ML/models/{self.LOAD_MODEL_FOLDER_NAME}/class_labels.txt") as f:
                self.class_labels = {
                    int(k): v for line in f for (
                        k, v) in [
                        line.strip().split(
                            None, 1)]}
            print(self.class_labels)

        # Camera
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

    def cv2pil(self, image):
        ''' OpenCV型 -> PIL型 '''
        new_image = image.copy()
        if new_image.ndim == 2:  # モノクロ
            pass
        elif new_image.shape[2] == 3:  # カラー
            new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
        elif new_image.shape[2] == 4:  # 透過
            new_image = cv2.cvtColor(new_image, cv2.COLOR_BGRA2RGBA)
        new_image = Image.fromarray(new_image)
        return new_image

    def get_contours(self, gray_img):
        """
        画像から輪郭を抽出する
        """
        contours, _ = cv2.findContours(
            gray_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def get_max_contour(self, contours):
        """
        与えられた輪郭の中で最大領域である輪郭とその面積を返す
        """
        if not contours:
            return None, 0

        max_contour = contours[0]
        max_area = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > max_area:
                max_contour = contour
                max_area = area
        return max_contour, max_area

    def mask_img_with_rgb(self, img, min_rgb, max_rgb):
        """
        画像をRGBに変換して、rgb(赤)でマスクする
        """
        min_r, min_g, min_b = min_rgb
        max_r, max_g, max_b = max_rgb
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        r = img_rgb[:, :, 0]
        g = img_rgb[:, :, 1]
        b = img_rgb[:, :, 2]
        mask = np.zeros(r.shape, np.uint8)
        mask[(min_r < r) & (r < max_r) & (min_g < g) & (
            g < max_g) & (min_b < b) & (b < max_b)] = 255
        return mask

    def mask_img_with_hsv(self, img, min_hue, max_hue, min_sat, max_sat):
        """
        画像をhsvに変換して、hue(色相)とsaturation(彩度)でマスクする
        """
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hue = img_hsv[:, :, 0]
        saturation = img_hsv[:, :, 1]
        mask = np.zeros(hue.shape, np.uint8)
        mask[(min_hue < hue) & (hue < max_hue) & (
            min_sat < saturation) & (saturation < max_sat)] = 255
        return mask

    def scan(self):
        ret, color_img = self.cap.read()
        origin_color_img = color_img.copy()
        if not ret:
            return 1

        mask = self.mask_img_with_rgb(
            color_img,
            self.WHITE_MIN_RGB,
            self.WHITE_MAX_RGB,)
        contours = self.get_contours(mask)
        max_contour, _ = self.get_max_contour(contours)
        if max_contour is None:
            return 1

        """四角形の枠をラップ"""
        cv2.drawContours(
            color_img,
            [max_contour],
            0,
            self.RED,
            self.LINE_WIDTH)

        # 単純な外接矩形
        x, y, w, h = cv2.boundingRect(max_contour)
        cv2.rectangle(color_img, (x, y), (x + w, y + h),
                      self.GREEN, self.LINE_WIDTH)

        # 傾きを考慮した外接矩形
        rect = cv2.minAreaRect(max_contour)
        box = np.int0(cv2.boxPoints(rect))
        cv2.drawContours(color_img, [box], 0, self.BLUE, self.LINE_WIDTH)
        cv2.imshow("color_img", color_img)

        """透視変換"""
        # 左上=box[0] 右上=box[1] 右下=box[2] 左下=box[3] (左上を0,0とする)
        src_points = np.array((box[0], box[1], box[2], box[3]), np.float32)
        bingo_card_size = (500, 500)
        dst_points = np.array([[0, 0], [bingo_card_size[0], 0], [bingo_card_size[0], bingo_card_size[1]], [
            0, bingo_card_size[1]]], np.float32)
        dst = np.array(bingo_card_size)
        M = cv2.getPerspectiveTransform(src_points, dst_points)
        warp = cv2.warpPerspective(origin_color_img, M, dst)
        bingo_card_crop = warp[50:450, 50:450]
        bingo_card_crop[80:400:80, :, :] = self.RED
        bingo_card_crop[:, 80:400:80, :] = self.RED
        cv2.imshow("bingo_card_crop", bingo_card_crop)
        return bingo_card_crop

    def convert_to_list(self, bingo_card_crop):
        """OCRを用いてリスト化"""
        bingo_card_list = [[None] * 5 for _ in range(5)]
        for row in range(5):
            for col in range(5):
                img = bingo_card_crop[row * 80:row *
                                      80 + 80, col * 80:col * 80 + 80]
                if row == 2 and col == 2:  # free
                    bingo_card_list[row][col] = "FREE"
                else:
                    if self.is_use_ocr:
                        num = self.ocr_tool.image_to_string(
                            self.cv2pil(img),
                            lang="eng",
                            builder=self.ocr_builder
                        )
                        num = num.replace(".", "").replace("-", "")
                    else:
                        img = cv2.resize(img, (40, 40))
                        num = self.class_labels[self.model.predict(img[None, ...])[
                            0].argmax()]

                    bingo_card_list[row][col] = 0 if num == "" else int(num)
        return bingo_card_list

    def quit(self):
        self.cap.release()
        cv2.destroyAllWindows()

    def run(self):
        while True:
            bingo_card_crop = self.scan()
            if bingo_card_crop is 1:
                continue

            if cv2.waitKey(1) & 0xFF == ord('c'):
                bingo_card_list = self.convert_to_list(bingo_card_crop)
                # self.print_bingo_card_list(bingo_card_list)
                is_bingo = Checker().check_bingo(bingo_card_list)
                if is_bingo:
                    if self.is_use_servo:
                        self.servo.set_angle(90)
                        time.sleep(self.servo_sleep_time)
                        self.servo.set_angle(0)
                    print("BINGO")
                else:
                    if self.is_use_servo:
                        self.servo.set_angle(-90)
                        time.sleep(self.servo_sleep_time)
                        self.servo.set_angle(0)
                    print("NOT BINGO")
                print()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.quit()
                break

    def print_bingo_card_list(self, bingo_card_list):
        for row in bingo_card_list:
            for col in row:
                print("{:^6s}".format(str(col)), end=" ")
            print()


if __name__ == "__main__":
    args = sys.argv
    is_use_servo = True if args[-1] == "use_servo" else False
    bingo_card_scanner = BingoCardScanner(is_use_servo=is_use_servo)
    bingo_card_scanner.run()
