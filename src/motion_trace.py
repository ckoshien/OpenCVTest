# coding: UTF-8

import time
import math
import cv2
import numpy as np

# ビデオデータ
VIDEO_DATA = "test.mp4"
outputFile="output.mp4"
# Esc キー
ESC_KEY = 0x1b
# モーションの残存期間(sec)
DURATION = 1.0
# 全体の方向を表示するラインの長さ
LINE_LENGTH_ALL = 60
# 座標毎の方向を表示するラインの長さ
LINE_LENGTH_GRID = 20
# 座標毎の方向を計算する間隔
GRID_WIDTH = 2
# 方向を表示するラインの丸の半径
CIRCLE_RADIUS = 2



# 表示ウィンドウの初期化
cv2.namedWindow("motion")
# ビデオデータの読み込み
outFourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
video = cv2.VideoCapture(VIDEO_DATA)
W = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(outputFile, outFourcc, 30.0,
                          (W, H))  # 出力先のファイルを開く

# 最初のフレームの読み込み
end_flag, frame_next = video.read()
height, width, channels = frame_next.shape
motion_history = np.zeros((height, width), np.float32)
frame_pre = frame_next.copy()

while(end_flag):
    # フレーム間の差分計算
    color_diff = cv2.absdiff(frame_next, frame_pre)

    # グレースケール変換
    gray_diff = cv2.cvtColor(color_diff, cv2.COLOR_BGR2GRAY)

    # ２値化
    retval, black_diff = cv2.threshold(gray_diff, 30, 1, cv2.THRESH_BINARY)

    # プロセッサ処理時間(sec)を取得
    proc_time = time.clock()

    # モーション履歴画像の更新
    cv2.motempl.updateMotionHistory(black_diff, motion_history, proc_time, DURATION)
    # 古いモーションの表示を経過時間に応じて薄くする
    hist_color = np.array(np.clip((motion_history - (proc_time - DURATION)) / DURATION, 0, 1) * 255, np.uint8)

    # グレースケール変換
    hist_gray = cv2.cvtColor(hist_color, cv2.COLOR_GRAY2BGR)
    
    # モーション履歴画像の変化方向の計算
    #   ※ orientationには各座標に対して変化方向の値（deg）が格納されます
    mask, orientation = cv2.motempl.calcMotionGradient(motion_history, 0.15, 0.001, apertureSize = 5)
    # 各座標の動きを緑色の線で描画
    # width_i = GRID_WIDTH
    # while width_i < width:
    #     height_i = GRID_WIDTH
    #     while height_i < height:
    #         cv2.circle(hist_gray, \
    #                    (width_i, height_i), \
    #                    CIRCLE_RADIUS, \
    #                    (0, 255, 0), \
    #                    2, \
    #                    16, \
    #                    0)
    #         angle_deg = orientation[height_i - 1][width_i - 1]
    #         if angle_deg > 0:
    #             print(height_i - 1,width_i - 1,angle_deg)
    #             angle_rad = math.radians(angle_deg)
    #             cv2.line(hist_gray, \
    #                     (width_i, height_i), \
    #                     (int(width_i + math.cos(angle_rad) * LINE_LENGTH_GRID), int(height_i + math.sin(angle_rad) * LINE_LENGTH_GRID)), \
    #                     (255, 0, 0), \
    #                     2, \
    #                     16, \
    #                     0)

    #         height_i += GRID_WIDTH

    #     width_i += GRID_WIDTH


    # 全体的なモーション方向を計算
    angle_deg = cv2.motempl.calcGlobalOrientation(orientation, mask, motion_history, proc_time, DURATION)

    # 全体の動きを黄色い線で描画
    # cv2.circle(hist_gray, \
    #            (int(width / 2), int(height / 2)), \
    #            CIRCLE_RADIUS, \
    #            (0, 215, 255), \
    #            2, \
    #            16, \
    #            0)
    # angle_rad = math.radians(angle_deg)
    # cv2.line(hist_gray, \
    #          (int(width / 2), int(height / 2)), \
    #          (int(width / 2 + math.cos(angle_rad) * LINE_LENGTH_ALL), int(height / 2 + math.sin(angle_rad) * LINE_LENGTH_ALL)), \
    #          (0, 215, 255), \
    #          2, \
    #          16, \
    #          0)
    dst=cv2.addWeighted(frame_next,1,hist_gray,0.5,0)

    # モーション画像を表示
    cv2.imshow("motion", dst)
    resizedImage = cv2.resize(dst, (W, H))
    out.write(resizedImage)

    # Escキー押下で終了
    if cv2.waitKey(20) == ESC_KEY:
        break

    # 次のフレームの読み込み
    frame_pre = frame_next.copy()
    end_flag, frame_next = video.read()
result = dst
#np.full((height, width, 3), 128, dtype=np.uint8)
for width_i in range(width - 1):
    for height_i in range(height - 1):
        if float(motion_history[height_i - 1][width_i - 1]) < 5.2 \
        and float(motion_history[height_i - 1][width_i - 1]) > 5.1 \
        and (height_i > 200 and height_i < 400) \
        and (width_i > 600 and width_i < 850):
            print(width_i, height_i, motion_history[height_i - 1][width_i - 1])
            cv2.rectangle(
                result,
                (600,200),
                (850,400),
                (0,255,0),
                3
            )
            cv2.circle(result, \
                    (width_i, height_i), \
                    CIRCLE_RADIUS, \
                    (0, 255, 0), \
                    2, \
                    16, \
                    0)
            cv2.imshow("result", result)
        height_i = height_i + GRID_WIDTH    
    width_i = width_i + GRID_WIDTH
cv2.imwrite('result.png', result)
# 終了処理
out.release()
cv2.destroyAllWindows()
video.release()