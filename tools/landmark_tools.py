import cv2


def read_points(file_name=None):
    """
    Read points from .pts file.
    """
    points = []
    with open(file_name) as file:
        line_count = 0
        for line in file:
            if "version" in line or "points" in line or "{" in line or "}" in line:
                continue
            else:
                loc_x, loc_y = line.strip().split(sep=" ")
                points.append([float(loc_x), float(loc_y)])
                line_count += 1
    return points


def draw_landmark_point(image, points):
    """
    Draw and show landmark point of the image.
    """
    for point in points:
        cv2.circle(image, (int(point[0]), int(
            point[1])), 3, (0, 255, 0), -1, cv2.LINE_AA)
    cv2.imshow("result", image)
    cv2.waitKey(0)


if __name__ == '__main__':
    points_read = read_points("E:\\BaiduNetdiskDownload\\300VW_Dataset_2015_12_14\\001\\annot\\000001.pts")
    image_read = cv2.imread("E:\\BaiduNetdiskDownload\\300VW_Dataset_2015_12_14\\001\\image\\1.jpg")
    draw_landmark_point(image_read, points_read)
