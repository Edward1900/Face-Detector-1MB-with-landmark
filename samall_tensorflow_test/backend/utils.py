import json
import numpy as np
import tensorflow as tf


def decode_regression_n(reg, image_size, feature_map_w_h_list, min_boxes,
                      center_variance, size_variance):
    priors = []
    for feature_map_w_h, min_box in zip(feature_map_w_h_list, min_boxes):
        xy_grid = np.meshgrid(range(feature_map_w_h[0]), range(feature_map_w_h[1]))
        xy_grid = np.add(xy_grid, 0.5)
        xy_grid[0, :, :] /= feature_map_w_h[0]
        xy_grid[1, :, :] /= feature_map_w_h[1]
        xy_grid = np.stack(xy_grid, axis=-1)
        xy_grid = np.tile(xy_grid, [1, 1, len(min_box)])
        xy_grid = np.reshape(xy_grid, (-1, 2))

        wh_grid = np.array(min_box) / np.array(image_size)[:, np.newaxis]
        wh_grid = np.tile(np.transpose(wh_grid), [np.product(feature_map_w_h), 1])

        prior = np.concatenate((xy_grid, wh_grid), axis=-1)
        priors.append(prior)

    priors = np.concatenate(priors, axis=0)
    print(f'priors nums:{priors.shape[0]}')

    #priors = tf.constant(priors, dtype=tf.float32, shape=priors.shape, name='priors')

    center_xy = reg[..., :2] * center_variance * priors[..., 2:] + priors[..., :2]
    center_wh = np.exp(reg[..., 2:] * size_variance) * priors[..., 2:]

    # center to corner
    start_xy = center_xy - center_wh / 2
    end_xy = center_xy + center_wh / 2

    loc = np.concatenate([start_xy, end_xy], axis=-1)
   # loc = tf.clip_by_value(loc, clip_value_min=0.0, clip_value_max=1.0)

    return loc

