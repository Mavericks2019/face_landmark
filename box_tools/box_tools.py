import cv2
from box_tools.landmark_tools import read_points
from dnn_rect_tools.face_rect_dectator import get_facebox


def get_square_box(box):
    """Get the square boxes which are ready for CNN from the boxes"""
    left_x = box[0]
    top_y = box[1]
    right_x = box[2]
    bottom_y = box[3]

    box_width = right_x - left_x
    box_height = bottom_y - top_y

    # Check if box is already a square. If not, make it a square.
    diff = box_height - box_width
    delta = int(abs(diff) / 2)

    if diff == 0:                   # Already a square.
        return box
    elif diff > 0:                  # Height > width, a slim box.
        left_x -= delta
        right_x += delta
        if diff % 2 == 1:
            right_x += 1
    else:                           # Width > height, a short box.
        top_y -= delta
        bottom_y += delta
        if diff % 2 == 1:
            bottom_y += 1

    # Make sure box is always square.
    assert ((right_x - left_x) == (bottom_y - top_y)), 'Box is not square.'
    return [left_x, top_y, right_x, bottom_y]


def move_box(box, offset):
    """Move the box to direction specified by offset"""
    left_x = box[0] + offset[0]
    top_y = box[1] + offset[1]
    right_x = box[2] + offset[0]
    bottom_y = box[3] + offset[1]
    return [left_x, top_y, right_x, bottom_y]


def fit_by_shifting(box, rows, cols):
    """Method 1: Try to move the box."""
    # Face box points.
    left_x = box[0]
    top_y = box[1]
    right_x = box[2]
    bottom_y = box[3]

    # Check if moving is possible.
    if right_x - left_x <= cols and bottom_y - top_y <= rows:
        if left_x < 0:                  # left edge crossed, move right.
            right_x += abs(left_x)
            left_x = 0
        if right_x > cols:              # right edge crossed, move left.
            left_x -= (right_x - cols)
            right_x = cols
        if top_y < 0:                   # top edge crossed, move down.
            bottom_y += abs(top_y)
            top_y = 0
        if bottom_y > rows:             # bottom edge crossed, move up.
            top_y -= (bottom_y - rows)
            bottom_y = rows

    return [left_x, top_y, right_x, bottom_y]


def fit_by_shrinking(box, rows, cols):
    """Method 2: Try to shrink the box."""
    # Face box points.
    left_x = box[0]
    top_y = box[1]
    right_x = box[2]
    bottom_y = box[3]

    # The first step would be get the interlaced area.
    if left_x < 0:                  # left edge crossed, set zero.
        left_x = 0
    if right_x > cols:              # right edge crossed, set max.
        right_x = cols
    if top_y < 0:                   # top edge crossed, set zero.
        top_y = 0
    if bottom_y > rows:             # bottom edge crossed, set max.
        bottom_y = rows

    # Then found out which is larger: the width or height. This will
    # be used to decide in which dimension the size would be shrunken.
    width = right_x - left_x
    height = bottom_y - top_y
    delta = abs(width - height)
    # Find out which dimension should be altered.
    if width > height:                  # x should be altered.
        if left_x != 0 and right_x != cols:     # shrink from center.
            left_x += int(delta / 2)
            right_x -= int(delta / 2) + delta % 2
        elif left_x == 0:                       # shrink from right.
            right_x -= delta
        else:                                   # shrink from left.
            left_x += delta
    else:                               # y should be altered.
        if top_y != 0 and bottom_y != rows:     # shrink from center.
            top_y += int(delta / 2) + delta % 2
            bottom_y -= int(delta / 2)
        elif top_y == 0:                        # shrink from bottom.
            bottom_y -= delta
        else:                                   # shrink from top.
            top_y += delta

    return [left_x, top_y, right_x, bottom_y]


def fit_box(box, image, points):
    """
    Try to fit the box, make sure it satisfy following conditions:
    - A square.
    - Inside the image.
    - Contains all the points.
    If all above failed, return None.
    """
    rows = image.shape[0]
    cols = image.shape[1]

    # First try to move the box.
    box_moved = fit_by_shifting(box, rows, cols)

    # If moving fails ,try to shrink.
    if box_is_valid(image, points, box_moved):
        return box_moved
    else:
        box_shrunken = fit_by_shrinking(box, rows, cols)

    # If shrink failed, return None
    if box_is_valid(image, points, box_shrunken):
        return box_shrunken

    # Finally, Worst situation.
    print("Fitting failed!")
    return None


def get_minimal_box(points):
    """
    Get the minimal bounding box of a group of points.
    The coordinates are also converted to int numbers.
    """
    min_x = int(min([point[0] for point in points]))
    max_x = int(max([point[0] for point in points]))
    min_y = int(min([point[1] for point in points]))
    max_y = int(max([point[1] for point in points]))
    return [min_x, min_y, max_x, max_y]


def points_in_box(points, box):
    """Check if box contains all the points"""
    minimal_box = get_minimal_box(points)
    return box[0] <= minimal_box[0] and \
        box[1] <= minimal_box[1] and \
        box[2] >= minimal_box[2] and \
        box[3] >= minimal_box[3]


def box_in_image(box, image):
    """Check if the box is in image"""
    rows = image.shape[0]
    cols = image.shape[1]
    return box[0] >= 0 and box[1] >= 0 and box[2] <= cols and box[3] <= rows


def box_is_valid(image, points, box):
    """Check if box is valid."""
    # Box contains all the points.
    points_is_in_box = points_in_box(points, box)
    if not points_is_in_box:
        print("points not in box!")
    # Box is in image.
    box_is_in_image = box_in_image(box, image)

    # Box is square.
    w_equal_h = (box[2] - box[0]) == (box[3] - box[1])

    # Return the result.
    return box_is_in_image and points_is_in_box and w_equal_h


def _get_positive_box(raw_boxes, points):
    for box in raw_boxes:
        # Move box down.
        diff_height_width = (box[3] - box[1]) - (box[2] - box[0])
        offset_y = int(abs(diff_height_width / 2))
        box_moved = move_box(box, [0, offset_y])

        # Make box square.
        square_box = get_square_box(box_moved)

        # Remove false positive boxes.
        if points_in_box(points, square_box):
            return square_box
    return None


def expand_box(square_box, scale_ratio=1.2):
    """Scale up the box"""
    assert (scale_ratio >= 1), "Scale ratio should be greater than 1."
    delta = int((square_box[2] - square_box[0]) * (scale_ratio - 1) / 2)
    left_x = square_box[0] - delta
    left_y = square_box[1] - delta
    right_x = square_box[2] + delta
    right_y = square_box[3] + delta
    return [left_x, left_y, right_x, right_y]


def get_valid_box(image, points):
    """
    Try to get a valid face box which meets the requirements.
    The function follows these steps:
        1. Try method 1, if failed:
        2. Try method 0, if failed:
        3. Return None
    """
    # Try method 1 first.

    # Try to get a positive box from face detection results.
    _, raw_boxes = get_facebox(image, threshold=0.5)
    positive_box = _get_positive_box(raw_boxes, points)
    if positive_box is not None:
        if box_in_image(positive_box, image) is True:
            return positive_box
        return fit_box(positive_box, image, points)

    # Method 1 failed, Method 0
    min_box = get_minimal_box(points)
    sqr_box = get_square_box(min_box)
    epd_box = expand_box(sqr_box)
    if box_in_image(epd_box, image) is True:
        return epd_box
    return fit_box(epd_box, image, points)


def show_points_and_sqr(image, box, points):
    label = "face"
    cv2.rectangle(image, (box[0], box[1]),
                  (box[2], box[3]), (0, 255, 0))
    label_size, base_line = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

    cv2.rectangle(image, (box[0], box[1] - label_size[1]),
                  (box[0] + label_size[0],
                   box[1] + base_line),
                  (0, 255, 0), cv2.FILLED)
    cv2.putText(image, label, (box[0], box[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    for point in points:
        cv2.circle(image, (int(point[0]), int(
            point[1])), 3, (0, 255, 0), -1, cv2.LINE_AA)
    cv2.imshow("result", image)
    cv2.waitKey(0)


if __name__ == '__main__':
    image1 = cv2.imread("E:\\BaiduNetdiskDownload\\300VW_Dataset_2015_12_14\\01_Indoor\\indoor_001.png")
    points1 = read_points("E:\\BaiduNetdiskDownload\\300VW_Dataset_2015_12_14\\01_Indoor\\indoor_001.pts")
    box1 = get_valid_box(image1, points1)
    show_points_and_sqr(image1, box1, points1)
