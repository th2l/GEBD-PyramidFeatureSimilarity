import math
import pickle

import keras.callbacks
import wandb
from scipy.signal import argrelmax
from tqdm import tqdm
import numpy as np
import tensorflow as tf
import keras.utils as kr_utils


def set_seed(seed=1):
    kr_utils.set_random_seed(seed)
    # tf.config.experimental.enable_op_determinism()


def set_gpu_growth():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only use the first GPU
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
            n_gpus = len(gpus)
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            n_gpus = 0
            print(e)
    else:
        n_gpus = 0

    return n_gpus


def set_mixed_precision(mixed_precision=True):
    if mixed_precision:
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        print('Mixed precision training')
    else:
        tf.keras.mixed_precision.set_global_policy('float32')
        print('Float32 training')


def get_bnd_signal(scores, fps, max_dur, sigma=1., score_threshold=0.1, min_threshold=0.25):
    scores_sigmoid = scores
    score_smooth = scores_sigmoid  # gaussian_filter1d(scores_sigmoid, sigma=sigma)
    cur_bnd = argrelmax(score_smooth, order=2)[0]
    cur_bnd = cur_bnd[cur_bnd < math.ceil(max_dur * fps)]
    win_size = 2
    cur_bnd_cos = []

    cur_bnd = cur_bnd[scores_sigmoid[cur_bnd] > score_threshold]

    cur_bnd = (1. * cur_bnd) / fps

    filter_time = np.logical_and(cur_bnd <= max_dur - min_threshold, cur_bnd >= min_threshold)
    # cur_bnd = cur_bnd[cur_bnd < max_dur]
    cur_bnd = cur_bnd[filter_time]

    return scores_sigmoid, cur_bnd.tolist(), cur_bnd_cos


def get_boundaries(bnd_scores, video_id, video_dur, cur_fps=5):
    bnd_dict = dict()
    score_dict = dict()
    bnd_dist_dict = dict()
    for idx in range(len(video_id)):
        cur_id = video_id[idx].decode('utf-8')

        sc_sig, cur_bnd, cur_dist = get_bnd_signal(bnd_scores[idx, :, 0], cur_fps, video_dur[idx], sigma=1)
        bnd_dict[cur_id] = cur_bnd
        score_dict[cur_id] = sc_sig
        bnd_dist_dict[cur_id] = cur_dist

    return bnd_dict


def challenge_eval_func(gt_path='', pred_path='', gt_dict=None, pred_dict=None, use_tqdm=False, verbose=False,
                        threshold=0.05):
    """
    https://github.com/StanLei52/GEBD/blob/main/Challenge_eval_Code/eval.py
    Latest commit 619e15c on Apr 8, 2021
    """
    # load GT files
    if gt_dict is None:
        if gt_path == '':
            gt_path = './k400_mr345_val_min_change_duration0.3.pkl'
        with open(gt_path, 'rb') as f:
            gt_dict = pickle.load(f, encoding='lartin1')

    # load output files
    if pred_dict is None:
        with open(pred_path, 'rb') as f:
            pred_dict = pickle.load(f, encoding='lartin1')

    # recall precision f1 for threshold 0.05(5%)

    tp_all = 0
    num_pos_all = 0
    num_det_all = 0

    if verbose:
        print('Evaluating F1@{}'.format(threshold))

    if use_tqdm:
        loop_list = tqdm(list(gt_dict.keys()))
    else:
        loop_list = list(gt_dict.keys())

    for vid_id in loop_list:

        # filter by avg_f1 score
        if gt_dict[vid_id]['f1_consis_avg'] < 0.3:
            continue

        if vid_id not in pred_dict.keys():
            num_pos_all += len(gt_dict[vid_id]['substages_timestamps'][0])
            continue

        # detected timestamps
        if isinstance(pred_dict[vid_id], dict):
            bdy_timestamps_det = pred_dict[vid_id]['bnd']
            pred_dict[vid_id]['dur'] = gt_dict[vid_id]['video_duration']
            pred_dict[vid_id]['fps'] = gt_dict[vid_id]['fps']
        else:
            bdy_timestamps_det = pred_dict[vid_id]

        # myfps = gt_dict[vid_id]['fps']
        my_dur = gt_dict[vid_id]['video_duration']
        ins_start = 0
        ins_end = my_dur

        # remove detected boundary outside the action instance
        tmp = []
        for det in bdy_timestamps_det:
            tmpdet = det + ins_start
            if ins_start <= tmpdet <= ins_end:
                tmp.append(tmpdet)
        bdy_timestamps_det = tmp
        if bdy_timestamps_det == []:
            num_pos_all += len(gt_dict[vid_id]['substages_timestamps'][0])
            continue
        num_det = len(bdy_timestamps_det)
        num_det_all += num_det

        # compare bdy_timestamps_det vs. each rater's annotation, pick the one leading the best f1 score
        bdy_timestamps_list_gt_allraters = gt_dict[vid_id]['substages_timestamps']
        f1_tmplist = np.zeros(len(bdy_timestamps_list_gt_allraters))
        tp_tmplist = np.zeros(len(bdy_timestamps_list_gt_allraters))
        num_pos_tmplist = np.zeros(len(bdy_timestamps_list_gt_allraters))

        for ann_idx in range(len(bdy_timestamps_list_gt_allraters)):
            bdy_timestamps_list_gt = bdy_timestamps_list_gt_allraters[ann_idx]
            num_pos = len(bdy_timestamps_list_gt)
            tp = 0
            offset_arr = np.zeros((len(bdy_timestamps_list_gt), len(bdy_timestamps_det)))
            for ann1_idx in range(len(bdy_timestamps_list_gt)):
                for ann2_idx in range(len(bdy_timestamps_det)):
                    offset_arr[ann1_idx, ann2_idx] = abs(
                        bdy_timestamps_list_gt[ann1_idx] - bdy_timestamps_det[ann2_idx])

            # print(offset_arr-threshold * my_dur)
            for ann1_idx in range(len(bdy_timestamps_list_gt)):
                if offset_arr.shape[1] == 0:
                    break
                min_idx = np.argmin(offset_arr[ann1_idx, :])
                if offset_arr[ann1_idx, min_idx] <= threshold * my_dur:
                    tp += 1
                    offset_arr = np.delete(offset_arr, min_idx, 1)

            num_pos_tmplist[ann_idx] = num_pos
            fn = num_pos - tp
            fp = num_det - tp
            if num_pos == 0:
                rec = 1
            else:
                rec = tp / (tp + fn)
            if (tp + fp) == 0:
                prec = 0
            else:
                prec = tp / (tp + fp)
            if (rec + prec) == 0:
                f1 = 0
            else:
                f1 = 2 * rec * prec / (rec + prec)
            tp_tmplist[ann_idx] = tp
            f1_tmplist[ann_idx] = f1

        ann_best = np.argmax(f1_tmplist)
        tp_all += tp_tmplist[ann_best]
        num_pos_all += num_pos_tmplist[ann_best]

        if isinstance(pred_dict[vid_id], dict):
            pred_dict[vid_id]['gt'] = bdy_timestamps_list_gt_allraters[ann_best]

    fn_all = num_pos_all - tp_all
    fp_all = num_det_all - tp_all
    if num_pos_all == 0:
        rec = 1.
    else:
        rec = tp_all / (tp_all + fn_all)
    if (tp_all + fp_all) == 0:
        prec = 0.
    else:
        prec = tp_all / (tp_all + fp_all)
    if (rec + prec) == 0:
        f1_final = 0.
    else:
        f1_final = 2 * rec * prec / (rec + prec)

    if verbose:
        print('Precision: {}   Recall: {}'.format(prec, rec))
        print('TP: {}.   FN: {}   . FP: {}'.format(tp_all, fn_all, fp_all))
        print('Num pos: {}.   Num det: {}'.format(num_pos_all, num_det_all))

    # with open('check.pkl', 'wb') as f:
    #     pickle.dump(pred_dict, f)
    return prec, rec, f1_final


class EvalCallback(tf.keras.callbacks.Callback):
    def __init__(self, val_data=None, cur_fps=5):
        self.val_data = val_data
        self.cur_fps = cur_fps
        self.best_weights = None
        self.best_f1 = -1.

    def on_epoch_end(self, epoch, logs=None):
        if self.val_data is not None:
            y_pred, y_vid_id, y_vid_dur = self.model.predict(self.val_data, verbose=0)
            pred_bnd = get_boundaries(y_pred, y_vid_id, y_vid_dur, cur_fps=self.cur_fps)
            scores = challenge_eval_func(pred_dict=pred_bnd, verbose=False)
            if scores[2] > self.best_f1:
                self.best_weights = self.model.get_weights()
                self.best_f1 = scores[2]
            eval_metric = {'val-f1': scores[2], 'val-prec': scores[0], 'val-rec': scores[1]}
            logs.update(eval_metric)

    def on_train_end(self, logs=None):
        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)


class LrLogger(tf.keras.callbacks.Callback):
    def __init__(self, log_freq=1, log_level='step'):
        """ Log learning rate at step/epoch every log_freq step/epoch, default on step"""
        super(LrLogger, self).__init__()
        self.log_freq = log_freq
        self.log_level = log_level

    def get_current_lr(self):
        if isinstance(self.model.optimizer.learning_rate, tf.keras.optimizers.schedules.LearningRateSchedule):
            current_lr = self.model.optimizer.learning_rate(self.model.optimizer.iterations)
        else:
            current_lr = self.model.optimizer.learning_rate

        return current_lr

    def on_batch_end(self, batch, logs=None):
        if self.log_level == 'step' and batch % self.log_freq == 0:
            current_lr = self.get_current_lr()
            logs.update({'lr': current_lr})

    def on_epoch_end(self, epoch, logs=None):
        if self.log_level == 'epoch':
            current_lr = self.get_current_lr()

            logs.update({'lr': current_lr})

    def on_train_end(self, logs=None):
        if self.log_level == 'step' and self.model.optimizer.iterations % self.log_freq != 0:
            current_lr = self.get_current_lr()
            logs.update({'lr': current_lr})


if __name__ == '__main__':
    for thresh in np.arange(0.05, 0.5, 0.05):
        scores = challenge_eval_func(pred_path='./submission_github.com_MCG-NJU_DDM.pkl', verbose=False,
                                     threshold=thresh)
        print(thresh, scores[-1])
    pass
