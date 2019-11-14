import os
import json
import cv2
from file_manager.dir_manager import DirManager, file_extension
from box_tools.boxs_tools import read_points, get_valid_box, show_points_and_sqr


def get_object_and_landmark_json(root_path):
    root_manager = DirManager(root_path)
    abs_bath_list = root_manager.get_file_abspath()
    for path in abs_bath_list:
        print(path)
        if file_extension(path) == ".pts":
            save_dict = {}
            pts_path = path
            image_path = os.path.splitext(path)[0] + ".jpg"
            json_path = os.path.splitext(path)[0] + ".json"
            image1 = cv2.imread(image_path)
            points1 = read_points(pts_path)
            box1 = get_valid_box(image1, points1)
            if not box1:
                print(path)
                dirty_data_list.append(path)
            # show_points_and_sqr(image1, box1, points1)
            save_dict["image_path"] = image_path
            save_dict["points"] = points1
            save_dict["box"] = box1
            with open(json_path, "w") as f:
                json.dump(save_dict, f)


if __name__ == '__main__':
    # get_object_and_landmark_json("E:\\BaiduNetdiskDownload\\300 Face in Wild dataset\\02_Outdoor")
    input_path = "E:\\BaiduNetdiskDownload\\300VW_Dataset_2015_12_14"
    dirty_data_path = "E:\\BaiduNetdiskDownload\\300VW_Dataset_2015_12_14\\dirty_data"
    dir_list = next(os.walk(input_path))
    root = dir_list[0]
    dir_list = dir_list[1]
    dirty_data_dict = {}
    for item in dir_list:
        dirty_data_list = []
        print(item)
        root_path = os.path.join(os.path.join(root, item), "images")
        get_object_and_landmark_json(root_path)
        if dirty_data_list:
            dirty_data_dict[item] = dirty_data_list
            with open(os.path.join(dirty_data_path, item + "dirty.json"), "w") as f:
                json.dump(dirty_data_dict, f)
