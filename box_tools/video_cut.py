import os
import shutil

import cv2


def process_video(input_video, output_path, skip_frame):
    cap = cv2.VideoCapture(input_video)
    expand_name = '.jpg'
    cnt = 0
    count = 0
    while True:
        ret, frame = cap.read()
        cnt += 1
        if cnt % skip_frame == 0:
            count += 1
        cv2.imwrite(os.path.join(output_path, "0" * (6-len(str(count))) + str(count) + expand_name), frame)
        if not ret:
            break


def process_video_and_move(input_video, output_path, pts_path, skip_frame, item_name):
    cap = cv2.VideoCapture(input_video)
    expand_name = '.jpg'
    pts_expand_name = '.pts'
    cnt = 0
    count = 0
    while True:
        ret, frame = cap.read()
        cnt += 1
        if cnt % skip_frame == 0:
            count += 1
        img_name = "0" * (6-len(str(count))) + str(count)
        cv2.imwrite(os.path.join(output_path, item_name + "_" + img_name + expand_name), frame)
        pts = os.path.join(pts_path, img_name + pts_expand_name)
        if not os.path.exists(pts):
            break
        shutil.copy(pts, os.path.join(output_path, item_name + "_" + img_name + pts_expand_name))
        if not ret:
            break


if __name__ == '__main__':
    # input_vid = "E:\\BaiduNetdiskDownload\\300VW_Dataset_2015_12_14\\001\\vid.avi"
    # output = "E:\\BaiduNetdiskDownload\\300VW_Dataset_2015_12_14\\001\\image"
    # skip = 1
    # process_video(input_vid, output, skip)
    input_path = "E:\\BaiduNetdiskDownload\\300VW_Dataset_2015_12_14"
    dir_list = next(os.walk(input_path))
    root = dir_list[0]
    dir_list = dir_list[1]
    for item in dir_list:
        print(item)
        root_path = os.path.join(root, item)
        video_path = os.path.join(root_path, "vid.avi")
        image_path = os.path.join(root_path, "images")
        pts = os.path.join(root_path, "annot")
        if not os.path.exists(image_path):
            os.mkdir(image_path)
        process_video_and_move(video_path, image_path, pts, 1, item)
