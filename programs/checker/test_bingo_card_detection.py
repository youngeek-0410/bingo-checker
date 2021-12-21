import cv2
import numpy as np


def get_contours(gray_img):
    """
    画像から輪郭を抽出する
    """
    contours, _ = cv2.findContours(
        gray_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def get_max_contour(contours):
    """
    与えられた輪郭の中で最大領域である輪郭とその面積を返す
    """
    max_contour = contours[0]
    max_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_contour = contour
            max_area = area
    return max_contour, max_area


def main():
    capture = cv2.VideoCapture(0)
    while(True):
        _, bingo_color_img = capture.read()
        cv2.imshow('bingo', bingo_color_img)
        bingo_gray_img = cv2.cvtColor(bingo_color_img, cv2.COLOR_BGR2GRAY)
        # threshold(sudoku_gray, それ以上を最大値にする閾値, 指定する最大値, cv2.THRESH_BINARY_INV)
        _, bingo_gray_inverse_img = cv2.threshold(
            bingo_gray_img, 150, 255, cv2.THRESH_BINARY_INV)

        # 最大輪郭抽出
        contours = get_contours(bingo_gray_inverse_img)
        max_contour, max_area = get_max_contour(contours)
        print(max_contour)

        # 四角形で近似
        epsilon = 0.1 * cv2.arcLength(contours, True)
        approx = cv2.approxPolyDP(contours, epsilon, True)

        merged_img = np.hstack(
            (bingo_color_img, bingo_gray_img, bingo_gray_inverse_img))
        cv2.imshow("merged_img", merged_img)

        # 正方形に透視変換
        # M = cv2.getPerspectiveTransform(pts1, pts2) # 透視変換行列
        # warp = cv2.warpPerspective(src, M, dst)
        # ret, warp_bin = cv2.threshold(warp, 150, 255, cv2.THRESH_BINARY)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            capture.release()
            break


if __name__ == "__main__":
    main()
