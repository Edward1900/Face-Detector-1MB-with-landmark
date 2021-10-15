import argparse
import sys

import cv2
import tensorflow as tf
import numpy as np
from backend.utils import decode_regression_n


def softmax(x):
    x_row_max = x.max(axis=-1)
    x_row_max = x_row_max.reshape(list(x.shape)[:-1]+[1])
    x = x - x_row_max
    x_exp = np.exp(x)
    x_exp_row_sum = x_exp.sum(axis=-1).reshape(list(x.shape)[:-1]+[1])
    softmax = x_exp / x_exp_row_sum
    return softmax


conf_threshold = 0.6
center_variance = 0.1
size_variance = 0.2

image_size = [320, 240]  # default input size 320*240
feature_map_wh_list = [[10, 8]]  # default feature map size
min_boxes = [[64, 96]]


def main():

    model = tf.keras.models.load_model("models/slim/face.h5")

    img = cv2.imread("imgs/t1.jpg")
    h, w, _ = img.shape
    img_resize = cv2.resize(img, (320, 240))
    img_resize = cv2.cvtColor(img_resize, cv2.COLOR_BGR2RGB)
    img_resize = img_resize - 127.0
    img_resize = img_resize / 128.0

    results = model.predict(np.expand_dims(img_resize, axis=0))  # result=[background,face,x1,y1,x2,y2]
    print(results.shape)
    reg,cls = results[:,:,:,0:8],results[:,:,:,8:]
    reg = reg.reshape([-1,4])
    print(reg.shape)
    cls = cls.reshape([-1, 2])
    print(cls.shape)

   # cls = np.Softmax(axis=-1)(cls)
    cls = softmax(cls)
    print(cls[:,1]>conf_threshold)
    cond = np.where(cls[:,1]>conf_threshold )
    loc = decode_regression_n(reg, image_size, feature_map_wh_list, min_boxes,
                            center_variance, size_variance)
    print(loc[cond])
    results = loc[cond]
    for result in results:
        start_x = int(result[0] * w)
        start_y = int(result[1] * h)
        end_x = int(result[2] * w)
        end_y = int(result[3] * h)

        cv2.rectangle(img, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)

    cv2.imwrite('imgs/test_output_res.jpg', img)


if __name__ == '__main__':
    main()
