import os

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
        cv2.imwrite(os.path.join(output_path, str(count) + expand_name), frame)
        if not ret:
            break


if __name__ == '__main__':
    input_vid = "E:\\BaiduNetdiskDownload\\300VW_Dataset_2015_12_14\\001\\vid.avi"
    output = "E:\\BaiduNetdiskDownload\\300VW_Dataset_2015_12_14\\001\\image"
    skip = 1
    process_video(input_vid, output, skip)
