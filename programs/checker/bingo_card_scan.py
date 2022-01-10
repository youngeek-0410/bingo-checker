import cv2
import numpy as np

from PIL import Image
import pyocr
import pyocr.builders


class Checker:
    """
    BINGO CHECKER

    Attributes:
    output_bingo_numbers (list): list of bingo numbers which already outputted in the game
    """

    def __init__(self):
        self.output_bingo_numbers = []

    def import_output_bingo_number(self) -> None:
        ls = ["FREE"]
        with open("./lottery_result.txt") as f:
            for line in f:
                ls.append(int(line))
        self.output_bingo_numbers = ls

    def check_bingo(self, bingo_card):
        self.print_bingo_card(bingo_card)
        self.import_output_bingo_number()
        self.print_output_bingo_numbers()
        binary_bingo_card = self.make_binary_bingo_card(bingo_card)
        self.print_bingo_card(binary_bingo_card)
        if self.check_binary_bingo_card(binary_bingo_card):
            return True
        return False

    def make_binary_bingo_card(self, bingo_card):
        binary_bingo_card = [[0] * 5 for _ in range(5)]
        for i in range(5):
            for j in range(5):
                if bingo_card[i][j] in self.output_bingo_numbers:
                    binary_bingo_card[i][j] = 1
        return binary_bingo_card

    def check_binary_bingo_card(self, binary_bingo_card):
        for i in range(5):
            if self.check_row(binary_bingo_card, i):
                return True
        for i in range(5):
            if self.check_column(binary_bingo_card, i):
                return True
        if self.check_diagonal(binary_bingo_card):
            return True
        if self.check_anti_diagonal(binary_bingo_card):
            return True
        return False

    def check_row(self, binary_bingo_card, row_index):
        row = binary_bingo_card[row_index]
        if row.count(1) == 5:
            return True
        return False

    def check_column(self, binary_bingo_card, column_index):
        column = [row[column_index] for row in binary_bingo_card]
        if column.count(1) == 5:
            return True
        return False

    def check_diagonal(self, binary_bingo_card):
        diagonal = [binary_bingo_card[i][i] for i in range(5)]
        if diagonal.count(1) == 5:
            return True
        return False

    def check_anti_diagonal(self, binary_bingo_card):
        anti_diagonal = [binary_bingo_card[i][4 - i] for i in range(5)]
        if anti_diagonal.count(1) == 5:
            return True
        return False

    def print_bingo_card(self, bingo_card_list):
        for row in bingo_card_list:
            for col in row:
                print("{:^6s}".format(str(col)), end=" ")
            print()
        print()

    def print_output_bingo_numbers(self):
        print(self.output_bingo_numbers)
        print()


class BingoCardScanner:
    LINE_WIDTH = 3
    RED = (0, 0, 255)
    GREEN = (0, 255, 0)
    BLUE = (255, 0, 0)

    # BINGOカードの外枠の白色
    WHITE_MIN_RGB = (130, 130, 130)
    WHITE_MAX_RGB = (255, 255, 255)

    def __init__(self, ocr_times=5):
        self.ocr_times = ocr_times

        # OCR
        tools = pyocr.get_available_tools()
        if len(tools) == 0:
            print("No OCR tool found")
            return
        self.ocr_tool = tools[0]
        self.ocr_builder = pyocr.builders.TextBuilder(tesseract_layout=6)
        self.ocr_builder.tesseract_configs.append("digits")

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
                if row == 2 and col == 2:  # free
                    bingo_card_list[row][col] = "FREE"
                else:
                    num = self.ocr_tool.image_to_string(
                        self.cv2pil(bingo_card_crop[row * 80:row *
                                                    80 + 80, col * 80:col * 80 + 80]),
                        lang="eng",
                        builder=self.ocr_builder
                    )
                    num = num.replace(".", "").replace("-", "")
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
                print(is_bingo)
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
    bingo_card_scanner = BingoCardScanner()
    bingo_card_scanner.run()
