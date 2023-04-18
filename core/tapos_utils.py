"""
Tapos dataset utils
"""
import os

# from model import get_filtered_output
from core.utils import get_bnd_signal, challenge_eval_func

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import json
from pathlib import Path
import contextlib2
import numpy as np

from tqdm import tqdm
import tensorflow as tf

from core.dataloader import open_sharded_output_tfrecords, video_features, int64_feature, float_feature, \
    bytes_feature
import pickle
from functools import partial


def create_tapos_video_tf_example(vid_id, vid_fps, vid_dur, vid_n_frames, substages_timestamps, target_fps,
                                  target_size=320,
                                  root_path='/mnt/SharedProject/Dataset/LOVEU_22/tapos'):
    # vid_id, vid_fps, vid_dur, substages_timestamps = vid_info_input
    # Convert video to mid fps (25fps) then resample to target fps
    mid_fps = 25
    n_frame_new = tf.cast(tf.math.ceil(vid_dur * target_fps), tf.int32)
    original_dur = vid_dur

    round_sec = tf.math.ceil(vid_dur)
    # Resample to mid fps
    base_frame_sec = tf.cast(tf.linspace(0, tf.cast(tf.math.ceil(vid_fps) - 1, tf.int32), mid_fps), tf.int32)
    # Resample to target fps
    base_frame_sec_target = tf.cast(tf.linspace(0, tf.cast(mid_fps - 1, tf.int32), target_fps), tf.int32)
    if target_fps != mid_fps:
        base_frame_idx = tf.gather(base_frame_sec, base_frame_sec_target)
    else:
        base_frame_idx = base_frame_sec_target

    # Get frame_idx base on target fps
    frame_idx = tf.TensorArray(tf.int32, size=tf.cast(round_sec, tf.int32))
    for idx in tf.range(round_sec, dtype=tf.int32):
        frame_idx = frame_idx.write(idx, base_frame_idx + idx * tf.cast(tf.math.ceil(vid_fps), tf.int32))

    frame_idx = frame_idx.concat()[:n_frame_new]

    max_dur = tf.cast(tf.math.ceil(round_sec / 10) * 10, tf.int32)

    target_n_frames = max_dur * target_fps
    video_frames = tf.TensorArray(tf.string, size=target_n_frames)
    for idx in tf.range(target_n_frames, dtype=tf.int32):
        # tf.autograph.experimental.set_loop_options(maximum_iterations=10)
        if idx >= n_frame_new or frame_idx[idx] >= vid_n_frames:
            img = tf.image.encode_jpeg(tf.zeros((256, 340, 3), dtype=tf.uint8))
        else:
            cur_frame_idx = frame_idx[idx]
            img_path = tf.strings.join(
                [root_path, f'v_{vid_id}',
                 'img_' + tf.strings.as_string(cur_frame_idx + 1, width=5, fill='0') + '.jpg'],
                separator='/')
            img = tf.io.read_file(img_path)

        video_frames = video_frames.write(idx, img)

    video_frames = video_frames.stack()

    # Resize to target size
    video_frames = tf.nest.map_structure(tf.stop_gradient, tf.map_fn(tf.image.decode_jpeg, video_frames,
                                                                     fn_output_signature=tf.uint8))
    video_frames = tf.image.convert_image_dtype(video_frames, dtype=tf.float32)
    video_frames = tf.image.resize_with_pad(video_frames, target_size, target_size)
    video_frames = tf.image.convert_image_dtype(video_frames, dtype=tf.uint8)
    video_frames = tf.nest.map_structure(tf.stop_gradient, tf.map_fn(tf.image.encode_jpeg, video_frames,
                                                                     fn_output_signature=tf.string))
    # End Resize to target size

    target_dur = 10
    cut_idx = tf.cast(target_fps * target_dur / 2, tf.int32)

    if target_n_frames > target_fps * target_dur:
        n_block = target_n_frames.numpy() // (target_fps * target_dur)
        video_frames_non_overlap = tf.reshape(video_frames, (n_block, target_fps * target_dur))
        video_frames_mid_overlap = tf.reshape(video_frames[cut_idx: target_n_frames - cut_idx],
                                              (n_block - 1, target_fps * target_dur))
        video_frames = tf.concat([video_frames_non_overlap, video_frames_mid_overlap], axis=0)
    else:
        n_block = 1
        video_frames = tf.expand_dims(video_frames, axis=0)

    feature_dict = {
        'image/height': int64_feature(target_size),
        'image/width': int64_feature(target_size),
        'num_frames': int64_feature(target_n_frames),
        'vid_dur': float_feature(target_dur),
        'original_dur': float_feature(original_dur),
        'current_fps': float_feature(target_fps),
    }

    tf_examples = []
    if n_block == 1:
        bnd_timestamps = substages_timestamps
        feature_dict.update({'vid_id': bytes_feature(vid_id.encode('utf-8')),
                             'bnd_timestamps': bytes_feature(bnd_timestamps.encode('utf-8')),
                             'video': video_features(tf.io.serialize_tensor(video_frames[0]))})
        tf_examples.append(tf.train.Example(features=tf.train.Features(feature=feature_dict)))
    else:
        if substages_timestamps == '':
            print('No boundary: ', vid_id)
            return []
        bnd_float = np.array([float(x) for x in substages_timestamps.split(',')])
        for idx in range(2 * n_block - 1):
            subtraction = 0
            if idx < n_block:
                sel_bnds = np.argwhere(np.logical_and(target_dur * idx < bnd_float, bnd_float < target_dur * (idx + 1)))
                subtraction = target_dur * idx
            else:
                # Set boundary by  10 second, start with 5
                sel_bnds = np.argwhere(
                    np.logical_and(target_dur * (idx - n_block) + target_dur / 2 < bnd_float,
                                   bnd_float < target_dur * (idx - n_block + 1) + target_dur / 2))
                subtraction = target_dur * (idx - n_block) + target_dur / 2

            if len(sel_bnds) == 0:
                continue
            bnd_timestamps = ','.join(str(x - subtraction) for x in bnd_float[sel_bnds.flatten()])

            feature_dict.update({'vid_id': bytes_feature(f'{vid_id}_{idx}'.encode('utf-8')),
                                 'bnd_timestamps': bytes_feature(bnd_timestamps.encode('utf-8')),
                                 'video': video_features(tf.io.serialize_tensor(video_frames[idx]))})

            tf_examples.append(tf.train.Example(features=tf.train.Features(feature=feature_dict)))

    return tf_examples


def create_tapos_tf_records(root_path, target_fps=25, target_size=320, split='train', num_shards=5):
    """
    Create tfrecords for current split
    :param root_path:
    :param split:
    :return:
    """

    tapos_json_path = f'{root_path}/tapos_annotation.json'
    anno_json = json.load(open(tapos_json_path, 'r'))

    vids = anno_json.keys()
    dict_data = dict()

    for vid in tqdm(vids):
        for instance in anno_json[vid]:
            if anno_json[vid][instance]['subset'] == split:
                pass
            else:
                continue
            dict_data[f'{vid}_{instance}'] = dict()

            tmp = instance.split('_')
            instance_start = float(tmp[1] + '.' + tmp[2])  # start at this shot
            instance_end = float(tmp[3] + '.' + tmp[4])  # end at this shot
            duration = instance_end - instance_start
            n_frames = int(anno_json[vid][instance]['total_frames'])
            bnds = anno_json[vid][instance]['substages'][1:-1]  # excluded the first and the end

            cur_fps = float(anno_json[vid][instance]['total_frames']) / duration
            bnds_sec = [np.array(bnds, dtype=float) / cur_fps]

            dict_data[f'{vid}_{instance}']['substages_timestamps'] = bnds_sec
            dict_data[f'{vid}_{instance}']['video_duration'] = duration
            dict_data[f'{vid}_{instance}']['fps'] = cur_fps
            dict_data[f'{vid}_{instance}']['f1_consis_avg'] = 1.

            dict_data[f'{vid}_{instance}']['new_fps'] = cur_fps
            dict_data[f'{vid}_{instance}']['new_dur'] = duration
            dict_data[f'{vid}_{instance}']['new_n_frame'] = n_frames

    with open(f'{root_path}/{split}.pkl', 'wb') as f:
        pickle.dump(dict_data, f)

    list_vid_id = sorted(list(dict_data.keys()))
    list_videos = []
    for vid_id in list_vid_id:
        if 'substages_timestamps' in dict_data[vid_id]:
            substages_timestamps = dict_data[vid_id]['substages_timestamps'][0]  # Only 1 annotator
            substages_timestamps = ','.join(str(x) for x in substages_timestamps)
        else:
            substages_timestamps = ''

        cur_vid_info = (
            vid_id, dict_data[vid_id]['new_fps'], dict_data[vid_id]['new_dur'], dict_data[vid_id]['new_n_frame'],
            substages_timestamps)
        list_videos.append(cur_vid_info)

    write_dir = Path('{}/tfrecords/{}'.format(root_path, split))
    write_dir.mkdir(exist_ok=True, parents=True)
    output_filebase = (write_dir / '{}.record'.format(split)).__str__()

    num_samples = 0
    with contextlib2.ExitStack() as tf_record_close_stack:
        output_tfrecords = open_sharded_output_tfrecords(tf_record_close_stack, output_filebase, num_shards)
        create_tf_example_func = partial(create_tapos_video_tf_example, target_fps=target_fps, target_size=target_size,
                                         root_path=f'{root_path}/rgb')
        for index, example in enumerate(tqdm(list_videos)):
            tf_examples = create_tf_example_func(*example)
            output_shard_index = index % num_shards
            for tf_example in tf_examples:
                output_tfrecords[output_shard_index].write(tf_example.SerializeToString())
                num_samples += 1

    print(f'Total samples in {split}: {num_samples}')


def load_tapos_clip(path, dur, fps, n_frames, target_fps=5, target_size=224):
    mid_fps = 25
    n_frame_new = tf.cast(tf.math.ceil(dur * target_fps), tf.int32)
    original_dur = dur

    round_sec = tf.math.ceil(dur)
    # Resample to mid fps
    base_frame_sec = tf.cast(tf.linspace(0, tf.cast(tf.math.ceil(fps) - 1, tf.int32), mid_fps), tf.int32)
    # Resample to target fps
    offset = tf.cast(tf.math.ceil(mid_fps / target_fps) - 1, tf.int32)

    base_frame_sec_target = tf.cast(tf.linspace(offset, tf.cast(mid_fps - 1, tf.int32), target_fps), tf.int32)
    base_frame_idx = tf.gather(base_frame_sec, base_frame_sec_target)

    # Get frame_idx base on target fps
    frame_idx = tf.TensorArray(tf.int32, size=tf.cast(round_sec, tf.int32))
    for idx in tf.range(round_sec, dtype=tf.int32):
        frame_idx = frame_idx.write(idx, base_frame_idx + idx * tf.cast(tf.math.ceil(fps), tf.int32))

    frame_idx = frame_idx.concat()[:n_frame_new]

    max_dur = tf.cast(tf.math.ceil(round_sec / 10) * 10, tf.int32)

    target_n_frames = max_dur * target_fps
    video_frames = tf.TensorArray(tf.string, size=target_n_frames)
    for idx in tf.range(target_n_frames, dtype=tf.int32):
        # tf.autograph.experimental.set_loop_options(maximum_iterations=10)
        if idx >= n_frame_new or frame_idx[idx] >= n_frames:
            img = tf.image.encode_jpeg(tf.zeros((256, 340, 3), dtype=tf.uint8))
        else:
            cur_frame_idx = frame_idx[idx]
            img_path = tf.strings.join(
                [path,
                 'img_' + tf.strings.as_string(cur_frame_idx + 1, width=5, fill='0') + '.jpg'],
                separator='/')
            img = tf.io.read_file(img_path)

        video_frames = video_frames.write(idx, img)

    video_frames = video_frames.stack()

    # Resize to target size
    video_frames = tf.nest.map_structure(tf.stop_gradient, tf.map_fn(tf.image.decode_jpeg, video_frames,
                                                                     fn_output_signature=tf.uint8))
    video_frames = tf.image.convert_image_dtype(video_frames, dtype=tf.float32)
    video_frames = tf.image.resize_with_pad(video_frames, target_size, target_size)
    video_frames = tf.image.convert_image_dtype(video_frames, dtype=tf.uint8)

    return video_frames


def tapos_evaluation(model, split='val'):
    root_path = '/mnt/SharedProject/Dataset/LOVEU_22/tapos/'
    tapos_json_path = f'{root_path}/tapos_annotation.json'
    anno_json = json.load(open(tapos_json_path, 'r'))

    vids = anno_json.keys()
    dict_data = dict()
    dict_pred = dict()

    with open('/mnt/SharedProject/Dataset/LOVEU_22/tapos/TAPOS_baseline_Shou2021/passfile.txt', 'r') as fd:
        passfile = fd.read().strip('\n').splitlines()
        keep_files = [x.replace(',', '_') for x in passfile]

    for vid in tqdm(vids):
        for instance in anno_json[vid]:
            if anno_json[vid][instance]['subset'] == split:
                pass
            else:
                continue

            tmp = instance.split('_')
            instance_start = float(tmp[1] + '.' + tmp[2])  # start at this shot
            instance_end = float(tmp[3] + '.' + tmp[4])  # end at this shot
            duration = instance_end - instance_start
            n_frames = int(anno_json[vid][instance]['total_frames'])
            bnds = anno_json[vid][instance]['substages'][1:-1]  # excluded the first and the end

            cur_fps = float(anno_json[vid][instance]['total_frames']) / duration
            bnds_sec = [np.array(bnds, dtype=float) / cur_fps]

            if f'{vid}_{instance}' not in keep_files:
                continue
            # if duration >= 200:  # with this, 0.5983
            #     continue
            dict_data[f'{vid}_{instance}'] = dict()
            dict_data[f'{vid}_{instance}']['substages_timestamps'] = bnds_sec
            dict_data[f'{vid}_{instance}']['video_duration'] = duration
            dict_data[f'{vid}_{instance}']['fps'] = cur_fps
            dict_data[f'{vid}_{instance}']['f1_consis_avg'] = 1.

            dict_data[f'{vid}_{instance}']['new_fps'] = cur_fps
            dict_data[f'{vid}_{instance}']['new_dur'] = duration
            dict_data[f'{vid}_{instance}']['new_n_frame'] = n_frames

            cur_video = load_tapos_clip(f'{root_path}rgb/v_{vid}_{instance}', duration, cur_fps, n_frames, target_fps=5,
                                        target_size=224)

            if cur_video.shape[0] > 50:
                if cur_video.shape[0] % 50 > 0:
                    print('Please check ', f'{vid}_{instance}')
                    continue
                frame_shape = cur_video.shape[1:]
                n_block = cur_video.shape[0] // 50
                non_overlap = tf.reshape(cur_video, (n_block, 50,) + frame_shape)
                mid_overlap = tf.reshape(cur_video[25:cur_video.shape[0] - 25], (n_block - 1, 50) + frame_shape)
                # cur_input = tf.concat([non_overlap, mid_overlap], axis=0)
                #
                # cur_output = model({'image': cur_input}, training=False)['outs']

                non_overlap_output = tf.reshape(model({'image': non_overlap}, training=False)['outs'], (-1, 1))
                mid_overlap_output = tf.reshape(model({'image': mid_overlap}, training=False)['outs'], (-1, 1))

                output = non_overlap_output + tf.pad(mid_overlap_output, paddings=((25, 25), (0, 0)))
                output = tf.expand_dims(output, axis=0)
            else:
                output = model({'image': tf.expand_dims(cur_video, axis=0)}, training=False)['outs']

            # output = get_filtered_output(output, size=5)

            # Get boundary from output
            sc_sig, cur_bnd, cur_dist = get_bnd_signal(output[0, :, 0].numpy(), 5, duration, sigma=1)
            dict_pred[f'{vid}_{instance}'] = cur_bnd

    scores_thresholds = []
    for thresh in np.arange(0.05, 0.51, 0.05):
        scores = challenge_eval_func(gt_dict=dict_data, pred_dict=dict_pred, verbose=False, threshold=thresh)
        write_str = '{},{:.4f},{:.4f}, F1: {:.4f}\n'.format(f'TAPOS-{split} - {thresh}', scores[0], scores[1], scores[2])
        print(write_str)
        scores_thresholds.append(scores)

    return scores_thresholds[0]


if __name__ == '__main__':
    tapos_root = '/mnt/SharedProject/Dataset/LOVEU_22/tapos'
    img_size = 224

    # tapos_evaluation(model=None, split='val')

    create_tapos_tf_records(tapos_root, target_fps=25, target_size=img_size, split='train', num_shards=100)
    # create_tapos_tf_records(tapos_root, target_fps=25, target_size=img_size, split='val')
