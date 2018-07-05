import numpy as np
import tensorflow as tf
import keras.backend as K

def dice_loss(training_mask, loss_weight, small_text_weight):
    def loss(y_true, y_pred):
        eps = 1e-5
        intersection = tf.reduce_sum(y_true * y_pred * tf.minimum(training_mask + small_text_weight, 1))
        union = tf.reduce_sum(y_true * tf.minimum(training_mask + small_text_weight, 1)) + tf.reduce_sum(y_pred * tf.minimum(training_mask + small_text_weight, 1)) + eps
        loss = 1. - (2. * intersection / union)
        return loss * loss_weight
    return loss

def rbox_loss(training_mask, small_text_weight, target_score_map):
    def loss(y_true, y_pred):
        # d1 -> top, d2->right, d3->bottom, d4->left
        d1_gt, d2_gt, d3_gt, d4_gt, theta_gt = tf.split(value=y_true, num_or_size_splits=5, axis=3)
        d1_pred, d2_pred, d3_pred, d4_pred, theta_pred = tf.split(value=y_pred, num_or_size_splits=5, axis=3)
        area_gt = (d1_gt + d3_gt) * (d2_gt + d4_gt)
        area_pred = (d1_pred + d3_pred) * (d2_pred + d4_pred)
        w_union = tf.minimum(d2_gt, d2_pred) + tf.minimum(d4_gt, d4_pred)
        h_union = tf.minimum(d1_gt, d1_pred) + tf.minimum(d3_gt, d3_pred)
        area_intersect = w_union * h_union
        area_union = area_gt + area_pred - area_intersect
        L_AABB = -tf.log((area_intersect + 1.0)/(area_union + 1.0))
        L_theta = 1 - tf.cos(theta_pred - theta_gt)
        L_g = L_AABB + 20 * L_theta
        return tf.reduce_mean(L_g * target_score_map * tf.minimum(training_mask + small_text_weight, 1))
    return loss
