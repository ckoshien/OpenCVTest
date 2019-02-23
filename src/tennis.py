import cv2
import sys
import numpy as np
#import os


def dilation(dilationSize, kernelSize, img):  # 膨張した画像にして返す
    kernel = np.ones((kernelSize, kernelSize), np.uint8)
    element = cv2.getStructuringElement(
        cv2.MORPH_RECT, (5 * dilationSize + 1, 5 * dilationSize + 1), (dilationSize, dilationSize))
    dilation_img = cv2.dilate(img, kernel, element)
    return dilation_img


def detect(gray_diff, thresh_diff=180, dilationSize=9, kernelSize=20):  # 一定面積以上の物体を検出
    retval, black_diff = cv2.threshold(
        gray_diff, thresh_diff, 255, cv2.THRESH_BINARY)  # 2値化
    dilation_img = dilation(dilationSize, kernelSize, black_diff)  # 膨張処理
    img = dilation_img.copy()
    #image,
    # 境界線検出
    contours, hierarchy = cv2.findContours(
        dilation_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )  

    ball_pos = []

    for i in range(len(contours)):  # 重心位置を計算
        count = len(contours[i])
        area = cv2.contourArea(contours[i])  # 面積計算
        x, y = 0.0, 0.0
        for j in range(count):
            x += contours[i][j][0][0]
            y += contours[i][j][0][1]

        x /= count
        y /= count
        x = int(x)
        y = int(y)
        ball_pos.append([x, y])

    return ball_pos, img


def displayCircle(image, ballList, thickness=5):
    for i in range(len(ballList)):
        x = int(ballList[i][0])
        y = int(ballList[i][1])
        cv2.circle(image, (x, y), 10, (0, 0, 255), thickness)
    return image


def resizeImage(image, w=2, h=2):
    height = image.shape[0]
    width = image.shape[1]
    resizedImage = cv2.resize(image, (int(width / w), int(height / h)))
    return resizedImage


def blackToColor(bImage):
    colorImage = np.array((bImage, bImage, bImage))
    colorImage = colorImage.transpose(1, 2, 0)
    return colorImage


def run(input_video_path, output_video_path):
    video = cv2.VideoCapture(input_video_path)  # videoファイルを読み込む
    #inFourcc = cv2.VideoWriter_fourcc(*'mp4v')
    outFourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

    if not video.isOpened():  # ファイルがオープンできない場合の処理.
        print("Could not open video")
        sys.exit()

    vidw = video.get(cv2.CAP_PROP_FRAME_WIDTH)
    vidh = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(vidw,vidh)
    out = cv2.VideoWriter(output_video_path, outFourcc, 10.0,
                          (int(vidw), int(vidh/2)))  # 出力先のファイルを開く

    ok, frame = video.read()  # 最初のフレームを読み込む
    if not ok:
        print('Cannot read video file')
        sys.exit()

    frame_pre = frame.copy()

    while True:
        ok, frame = video.read()  # フレームを読み込む
        if not ok:
            break
        frame_next = frame.copy()
        color_diff = cv2.absdiff(frame_next, frame_pre)  # フレーム間の差分計算
        video.read()[1]
        video.read()[1]
        frame4 = video.read()[1]
        #frame4 = frame.copy()
        color_diff2 = cv2.absdiff(frame4, frame_next)
        diff = cv2.bitwise_and(color_diff,color_diff2)
        #print(diff)
        #gray_diff = cv2.cvtColor(color_diff, cv2.COLOR_BGR2GRAY)  # グレースケール変換
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        retval, black_diff = cv2.threshold(
            gray_diff, 30, 255, cv2.THRESH_BINARY)

        ball, dilation_img = detect(gray_diff)

        frame = displayCircle(frame, ball, 2)  # 丸で加工
        cImage = blackToColor(dilation_img)  # 2値化画像をカラーの配列サイズと同じにする
        im1 = resizeImage(frame, 2, 2)
        im2 = resizeImage(cImage, 2, 2)
        im_h = cv2.hconcat([im1, im2])  # 画像を横方向に連結

        cv2.imshow("Tracking", im_h)  # フレームを画面表示
        out.write(im_h)
        #print(im_h)

        frame_pre = frame_next.copy()  # 次のフレームの読み込み

        k = cv2.waitKey(1) & 0xff  # ESCを押したら中止
        if k == 27:
            break
    video.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    inputFile="test.mp4"
    #inputFile='1.mp4'
    outputFile="output.mp4"
    run(inputFile, outputFile)