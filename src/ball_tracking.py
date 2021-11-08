import glob
import re
import cv2
import numpy as np


VIDEOPATH = "test.mp4"
IMAGEPATH = "E:/GitHub/local/OpenCVTest/src/"
TEMPLATEPATH = "template.jpeg"


def save_frames(video_path, image_dir):
    """
    動画からフレームの画像を抽出
    """
    cap = cv2.VideoCapture(video_path)
    digit = len(str(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))
    n = 0
    while True:
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(IMAGEPATH + "original/frame_"+ str(n) + ".jpeg", frame)
            n += 1
        else:
            return


def do_grayscale(image_path):
    """
    画像をグレースケール化
    """
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    save_image(image_path, "gray", gray)


def do_binarization(image_path):
    """
    画像を2値化
    """
    img = cv2.imread(image_path)
    ret, img_thresh = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
    save_image(image_path, "binary", img_thresh)


def do_backgroundsub():
    """
    背景差分を行う
    """
    
    img_list = glob.glob(IMAGEPATH + "binary/frame*.jpeg")
    num = lambda val: int(re.sub("\D","",val))
    sorted(img_list,key=(num))
    source = img_list[0]
    for path in img_list:
        diff = cv2.absdiff(cv2.imread(source),cv2.imread(path))
        source = path
        save_image(path, "bgsub", diff)


def do_template_matching():
    """
    テンプレート画像とフレーム画像でテンプレートマッチングを行う
    """
    template_img = cv2.imread(IMAGEPATH + "binary/" + TEMPLATEPATH)
    img_list = glob.glob(IMAGEPATH + "bgsub/frame*.jpeg")
    num = lambda val: int(re.sub("\D","",val))
    sorted(img_list,key=(num))
    location_list = []
    for path in img_list:
        result = cv2.matchTemplate(cv2.imread(path), template_img, cv2.TM_CCOEFF_NORMED)
        result = np.where(result >= 0.8, result, 0)
        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(result)
        print(maxLoc)
        location_list.append(maxLoc)
    return location_list

def draw_rectangle(location_list):
    """
    マッチング結果を画像に描画する
    """
    source = cv2.imread(IMAGEPATH + "original/frame_0.jpeg")
    cv2.imwrite(IMAGEPATH + "result.jpeg",source)
    source = cv2.imread(IMAGEPATH + "result.jpeg")
    for loc in location_list:
        lx, ly, rx, ry = loc[0] - 10, loc[1] - 10, loc[0] + 10, loc[1] + 10
        img = cv2.rectangle(source, (lx, ly), (rx, ry), (0, 255, 0), 3)
        cv2.imwrite(IMAGEPATH + "result.jpeg",img)

def save_image(img_path, dir, img):
    """
    画像を保存する
    img_path : 画像のパス
    dir : ディレクトリ名
    img : 画像データ
    """
    file_name = img_path.replace("\\","/").split(".")[0].split("/")[-1]
    cv2.imwrite("{}{}/{}.{}".format(IMAGEPATH, dir, file_name,"jpeg"), img)


if __name__=="__main__":
    # ①動画をフレームごとに分割
    save_frames(VIDEOPATH,IMAGEPATH)
    # ②テンプレート画像とフレーム画像をグレースケール化
    do_grayscale(IMAGEPATH + TEMPLATEPATH)
    for path in glob.glob(IMAGEPATH + "original/*.jpeg"):
        do_grayscale(path)
    # ③テンプレート画像とフレーム画像の2値化
    for path in glob.glob(IMAGEPATH + "gray/*.jpeg"):
        do_binarization(path)
    # ④背景差分を行う
    do_backgroundsub()
    # ⑤テンプレートマッチングを行う
    location_list = do_template_matching()
    # ⑥マッチングした座標を投影
    draw_rectangle(location_list)