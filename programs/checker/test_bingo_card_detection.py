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


def mask_img_with_rgb(img, min_rgb, max_rgb):
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


def main():
    LINE_WIDTH = 3
    RED = (0, 0, 255)
    GREEN = (0, 255, 0)
    BLUE = (255, 0, 0)

    WHITE_MIN_RGB = (130, 130, 130)
    WHITE_MAX_RGB = (255, 255, 255)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    while(True):
        ret, color_img = cap.read()
        origin_color_img = color_img.copy()
        if not ret:
            continue

        mask = mask_img_with_rgb(
            color_img,
            WHITE_MIN_RGB,
            WHITE_MAX_RGB,)
        contours = get_contours(mask)
        max_contour, _ = get_max_contour(contours)
        if max_contour is None:
            continue

        """四角形の枠をラップ"""
        cv2.drawContours(color_img, [max_contour], 0, RED, LINE_WIDTH)

        # 単純な外接矩形
        x, y, w, h = cv2.boundingRect(max_contour)
        cv2.rectangle(color_img, (x, y), (x + w, y + h), GREEN, LINE_WIDTH)

        # 傾きを考慮した外接矩形
        rect = cv2.minAreaRect(max_contour)
        box = np.int0(cv2.boxPoints(rect))
        cv2.drawContours(color_img, [box], 0, BLUE, LINE_WIDTH)
        cv2.imshow("color_img", color_img)

        """透視変換"""
        # 左上=box[0] 右上=box[1] 右下=box[2] 左下=box[3] (左上を0,0とする)
        src_points = np.array((box[0], box[1], box[2], box[3]), np.float32)
        bingo_card_size = (400, 500)
        dst_points = np.array([[0, 0], [bingo_card_size[0], 0], [bingo_card_size[0], bingo_card_size[1]], [
            0, bingo_card_size[1]]], np.float32)
        dst = np.array(bingo_card_size)
        M = cv2.getPerspectiveTransform(src_points, dst_points)
        warp = cv2.warpPerspective(origin_color_img, M, dst)
        cv2.imshow("bingo_card", warp)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            break


if __name__ == "__main__":
    main()
