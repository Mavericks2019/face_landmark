import cv2
from cv2 import dnn

WIDTH = 300
HEIGHT = 300
PROTOTXT = 'networks\\deploy.prototxt'
MODEL = 'networks\\res10_300x300_ssd_iter_140000.caffemodel'

NET = dnn.readNetFromCaffe(PROTOTXT, MODEL)


def get_facebox(image=None, threshold=0.5):
    """
    Get the bounding box of faces in image.
    """
    rows = image.shape[0]
    cols = image.shape[1]

    confidences = []
    faceboxes = []

    NET.setInput(dnn.blobFromImage(
        image, 1.0, (WIDTH, HEIGHT), (104.0, 177.0, 123.0), False, False))
    detections = NET.forward()

    for result in detections[0, 0, :, :]:
        confidence = result[2]
        if confidence > threshold:
            x_left_bottom = int(result[3] * cols)
            y_left_bottom = int(result[4] * rows)
            x_right_top = int(result[5] * cols)
            y_right_top = int(result[6] * rows)
            confidences.append(confidence)
            faceboxes.append(
                [x_left_bottom, y_left_bottom, x_right_top, y_right_top])
    return confidences, faceboxes


def draw_result(image, confidences, faceboxes):
    """Draw the detection result on image"""
    for result in zip(confidences, faceboxes):
        conf = result[0]
        facebox = result[1]

        cv2.rectangle(image, (facebox[0], facebox[1]),
                      (facebox[2], facebox[3]), (0, 255, 0))
        label = "face: %.4f" % conf
        label_size, base_line = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        cv2.rectangle(image, (facebox[0], facebox[1] - label_size[1]),
                      (facebox[0] + label_size[0],
                       facebox[1] + base_line),
                      (0, 255, 0), cv2.FILLED)
        cv2.putText(image, label, (facebox[0], facebox[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
        cv2.imshow("image", image)
        cv2.imwrite("ll.jpg", image)
        cv2.waitKey(0)


if __name__ == '__main__':
    image = cv2.imread("E:\\BaiduNetdiskDownload\\300VW_Dataset_2015_12_14\\001\\image\\ll.jpg")
    confidences, faceboxes = get_facebox(image)
    draw_result(image, confidences, faceboxes)
