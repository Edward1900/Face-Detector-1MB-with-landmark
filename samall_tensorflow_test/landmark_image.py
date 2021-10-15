import argparse
import sys
import os
import cv2
import tensorflow as tf
import numpy as np




class landmark(object):
    def __init__(self):
        self.PATH_TO_CKPT = './models/landmark/landmark_pb.pb'
        self.sess, self.image_tensor, self.output = self._load_model()
        self.img_size_w = 48
        self.img_size_h = 48

    def _load_model(self):
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()

            with tf.gfile.GFile(self.PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
                sess = tf.Session(graph=detection_graph)
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_tensor = detection_graph.get_tensor_by_name('Placeholder:0')

                output = detection_graph.get_tensor_by_name('MobileNet/conv_6_1:0')

                return sess,image_tensor,output
    def detect(self, img):
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.img_size_w, self.img_size_h))
        image = image.astype('float32')
        image = (image -127.5) * 0.0078125
        image_np_expanded = np.expand_dims(image, axis=0)
        #image_np_expanded = image_np_expanded.astype('float32')
        # Actual detection.
        resault = self.sess.run(
            self.output,
            feed_dict={self.image_tensor: image_np_expanded})
        return resault[0]


landmark= landmark()

w, h = 48, 48
img_src = cv2.imread('imgs/t2.jpg')
img_src = cv2.resize(img_src, (w, h))
logits1 = landmark.detect(img_src)
 # 人脸关键点回归测试
landmark_pred = logits1
print(landmark_pred)
landmark_pred = np.reshape(landmark_pred, (1, 5, 2))
landmark_pred_x = landmark_pred[0, :, 0] * 48
landmark_pred_y = landmark_pred[0, :, 1] * 48

print(landmark_pred_x)
print(landmark_pred_y)

for i in range(len(landmark_pred_x)):
   cv2.circle(img_src, (int(landmark_pred_x[i]), int(landmark_pred_y[i])), 2, (255, 0, 255))
img_src = cv2.resize(img_src, (112, 112))
cv2.imwrite("imgs/L.jpg", img_src)




