"""
Author: Van Thong Huynh
Affiliation: Dept. of AI Convergence, Chonnam Nat'l Univ.
"""
import pathlib
import pickle
from functools import partial

import numpy as np
import tensorflow as tf
import glob, os
import contextlib2
from tqdm import tqdm

try:
    from utils import set_gpu_growth

    set_gpu_growth()
    from backbone import ResNetBackbone

    backbone_model = ResNetBackbone()
except:
    pass

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def video_features(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.numpy()]))


def open_sharded_output_tfrecords(exit_stack, base_path, num_shards):
    """Opens all TFRecord shards for writing and adds them to an exit stack.
    Source:
    https://github.com/tensorflow/models/blob/master/research/object_detection/dataset_tools/tf_record_creation_util.py
    Args:
      exit_stack: A context2.ExitStack used to automatically closed the TFRecords
        opened in this function.
      base_path: The base path for all shards
      num_shards: The number of shards
    Returns:
      The list of opened TFRecords. Position k in the list corresponds to shard k.
    """
    tf_record_output_filenames = [
        '{}-{:05d}-of-{:05d}'.format(base_path, idx, num_shards)
        for idx in range(num_shards)
    ]

    options = tf.io.TFRecordOptions('')  # GZIP

    tfrecords = [
        exit_stack.enter_context(tf.io.TFRecordWriter(file_name, options))
        for file_name in tf_record_output_filenames
    ]

    return tfrecords


@tf.function
def get_backbone_prediction(inputs):
    return backbone_model(inputs, training=False)


def create_gebd_video_tf_example(vid_id, vid_fps, vid_dur, vid_n_frames, substages_timestamps, target_fps,
                                 target_size=320,
                                 root_path='/mnt/SharedProject/Dataset/LOVEU_22/gebd/frames'):
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

    max_dur = 10

    target_n_frames = max_dur * target_fps
    video_frames = tf.TensorArray(tf.string, size=target_n_frames)
    for idx in tf.range(target_n_frames, dtype=tf.int32):
        # tf.autograph.experimental.set_loop_options(maximum_iterations=10)
        if idx >= n_frame_new or frame_idx[idx] >= vid_n_frames:
            img = tf.image.encode_jpeg(tf.zeros((320, 320, 3), dtype=tf.uint8))
        else:
            cur_frame_idx = frame_idx[idx]
            img_path = tf.strings.join(
                [root_path, vid_id, tf.strings.as_string(cur_frame_idx, width=5, fill='0') + '.jpg'],
                separator='/')
            img = tf.io.read_file(img_path)

        video_frames = video_frames.write(idx, img)

    video_frames = video_frames.stack()

    if target_size != 320:
        video_frames = tf.nest.map_structure(tf.stop_gradient, tf.map_fn(tf.image.decode_jpeg, video_frames,
                                                                         fn_output_signature=tf.uint8))
        video_frames = tf.image.convert_image_dtype(video_frames, dtype=tf.float32)
        video_frames = tf.image.resize(video_frames, [target_size, target_size])
        video_frames = tf.image.convert_image_dtype(video_frames, dtype=tf.uint8)
        # video_frames = tf.nest.map_structure(tf.stop_gradient, tf.map_fn(tf.image.encode_jpeg, video_frames,
        #                                                                  fn_output_signature=tf.string))

    video_frames = tf.image.convert_image_dtype(video_frames, dtype=tf.float32) * 2. - 1.
    video_frames = get_backbone_prediction(video_frames)
    feature_dict = {ky: video_features(tf.io.serialize_tensor(video_frames[ky])) for ky in video_frames}

    bnd_timestamps = substages_timestamps
    feature_dict.update({
        'image/height': int64_feature(target_size),
        'image/width': int64_feature(target_size),
        'num_frames': int64_feature(target_n_frames),
        'vid_dur': float_feature(max_dur),
        'original_dur': float_feature(original_dur),
        'current_fps': float_feature(target_fps),
        'vid_id': bytes_feature(vid_id.encode('utf-8')),
        'bnd_timestamps': bytes_feature(bnd_timestamps.encode('utf-8')),
    })
    tf_example = tf.train.Example(features=tf.train.Features(feature=feature_dict))

    return tf_example


def create_tf_records(root_path, data_root, target_fps=25, target_size=320, split='train', num_shards=5):
    """
    Create tfrecords for current split
    :param root_path:
    :param data_root:
    :param split:
    :return:
    """
    with open(f'{root_path}export/{split}_data.pkl', 'rb') as f:
        dict_data = pickle.load(f, encoding='latin1')

    list_vid_id = sorted(list(dict_data.keys()))
    list_videos = []
    for vid_id in list_vid_id:
        if 'substages_timestamps' in dict_data[vid_id]:
            substages_timestamps = dict_data[vid_id]['substages_timestamps'][np.argmax(dict_data[vid_id]['f1_consis'])]
            substages_timestamps = ','.join(str(x) for x in substages_timestamps)
        else:
            substages_timestamps = ''

        cur_vid_info = (
            vid_id, dict_data[vid_id]['new_fps'], dict_data[vid_id]['new_dur'], dict_data[vid_id]['new_n_frame'],
            substages_timestamps)
        list_videos.append(cur_vid_info)

    write_dir = pathlib.Path('{}/../tfrecords_v2/{}'.format(data_root, split))
    write_dir.mkdir(exist_ok=True, parents=True)
    output_filebase = (write_dir / '{}.record'.format(split)).__str__()

    with contextlib2.ExitStack() as tf_record_close_stack:
        output_tfrecords = open_sharded_output_tfrecords(tf_record_close_stack, output_filebase, num_shards)
        create_tf_example_func = partial(create_gebd_video_tf_example, target_fps=target_fps, target_size=target_size,
                                         root_path=f'{data_root}/{split}')
        for index, example in enumerate(tqdm(list_videos)):
            tf_example = create_tf_example_func(*example)
            output_shard_index = index % num_shards
            output_tfrecords[output_shard_index].write(tf_example.SerializeToString())


def parse_gebd_tf_record(sequence_example, target_fps=8, target_size=320, random_offset=False):
    feature_description = {'image/height': tf.io.FixedLenFeature([], tf.int64),
                           'image/width': tf.io.FixedLenFeature([], tf.int64),
                           'num_frames': tf.io.FixedLenFeature([], tf.int64),
                           'vid_dur': tf.io.FixedLenFeature([], tf.float32),
                           'original_dur': tf.io.FixedLenFeature([], tf.float32),
                           'current_fps': tf.io.FixedLenFeature([], tf.float32),
                           'vid_id': tf.io.FixedLenFeature([], tf.string),
                           'bnd_timestamps': tf.io.FixedLenFeature([], tf.string),
                           'video': tf.io.FixedLenFeature([], tf.string),
                           # 'last': tf.io.FixedLenFeature([], tf.string),
                           # 'conv2': tf.io.FixedLenFeature([], tf.string),
                           # 'conv3': tf.io.FixedLenFeature([], tf.string),
                           # 'conv4': tf.io.FixedLenFeature([], tf.string),
                           }

    example_info = tf.io.parse_single_example(sequence_example, feature_description)
    # For image
    example_video = tf.io.parse_tensor(example_info['video'], out_type=tf.string)

    # For extracted features
    # feat_last = tf.io.parse_tensor(example_info['last'], out_type=tf.float32)
    # feat_conv2 = tf.io.parse_tensor(example_info['conv2'], out_type=tf.float32)
    # feat_conv3 = tf.io.parse_tensor(example_info['conv3'], out_type=tf.float32)
    # feat_conv4 = tf.io.parse_tensor(example_info['conv4'], out_type=tf.float32)

    curr_fps = example_info['current_fps']
    n_frames = tf.cast(example_info['num_frames'], tf.int32)

    if target_fps < curr_fps:
        # Resampling to lower fps
        n_frames = tf.cast(tf.math.ceil(example_info['vid_dur'] * target_fps), tf.int32)
        round_sec = tf.math.ceil(example_info['vid_dur'])

        offset = tf.cast(tf.math.ceil(curr_fps / target_fps) - 1, tf.int32)

        base_frame_idx = tf.cast(
            tf.round(tf.linspace(offset, tf.cast(tf.math.ceil(curr_fps) - 1, tf.int32), target_fps)),
            tf.int32)

        frame_idx = tf.TensorArray(tf.int32, size=tf.cast(round_sec, tf.int32))
        for idx in tf.range(round_sec, dtype=tf.int32):
            frame_idx = frame_idx.write(idx, base_frame_idx + idx * tf.cast(tf.math.ceil(curr_fps), tf.int32))

        frame_idx = frame_idx.concat()[:n_frames]

        if random_offset:
            frame_idx = frame_idx - tf.random.uniform(shape=[n_frames, ], maxval=offset, dtype=tf.int32)
        # For image
        example_video = tf.gather(example_video, frame_idx)

        # For extracted features
        # feat_last = tf.gather(feat_last, frame_idx)
        # feat_conv2 = tf.gather(feat_conv2, frame_idx)
        # feat_conv3 = tf.gather(feat_conv3, frame_idx)
        # feat_conv4 = tf.gather(feat_conv4, frame_idx)

    video_frames = tf.nest.map_structure(tf.stop_gradient,
                                         tf.map_fn(tf.image.decode_jpeg, example_video, fn_output_signature=tf.uint8))

    if tf.strings.length(example_info['bnd_timestamps']) > 0:
        bnd_idx_tmp = tf.cast(
            tf.strings.to_number(tf.strings.split([example_info['bnd_timestamps']], ',')) * target_fps,
            tf.int32)

        bnd_idx = bnd_idx_tmp  # tf.concat([bnd_idx_tmp-1, bnd_idx_tmp, bnd_idx_tmp+1], axis=-1)
        anno = tf.scatter_nd(tf.reshape(bnd_idx, (-1, 1)),
                             tf.reshape(tf.ones_like(bnd_idx), [-1]),
                             tf.reshape(n_frames, (1,))
                             )
    else:
        anno = tf.zeros([n_frames], dtype=tf.int32)

    anno = tf.cast(anno, tf.float32)

    # sample_weights = tf.cast(tf.random.categorical(tf.math.log([[0.3, 0.7]]), tf.size(anno)), dtype=tf.float32)
    # sample_weights = tf.squeeze(sample_weights, axis=0)
    # sample_weights = tf.expand_dims(tf.reduce_max([sample_weights, anno], axis=0), axis=-1)

    video_frames = tf.ensure_shape(video_frames, [10 * target_fps, target_size, target_size, 3])
    output_dict = {
        # "feat_last": feat_last,
        # "feat_conv2": feat_conv2,
        # "feat_conv3": feat_conv3,
        # "feat_conv4": feat_conv4,
        "image": video_frames,
        "num_frames": n_frames,
        "vid_id": example_info['vid_id'],
        "vid_dur": example_info['vid_dur'],
        'original_dur': example_info['original_dur'],
        'curr_fps': target_fps,
        # 'vid_bnd': anno
    }

    return output_dict, anno


def augmentation(inputs_dict):
    video = inputs_dict['image']

    pass


def get_dataset(data_root, target_fps=8, target_size=320, split='train', batch_size=32, n_gpus=1):
    if split != 'train':
        tfrecord_pattern_name = f'{split}.record-?????-of-00005'
    else:
        tfrecord_pattern_name = f'{split}.record-?????-of-00100'

    list_files = list((pathlib.Path(data_root) / f'tfrecords/{split}').glob(tfrecord_pattern_name))
    if split == 'train':
        list_files_str = [x.__str__() for x in list_files]
        print('Num tfrecord train: ', len(list_files_str))

        list_files = tf.data.Dataset.from_tensor_slices(list_files_str).cache().shuffle(len(list_files_str),
                                                                                        reshuffle_each_iteration=True)
        # list_files = list_files.repeat()

    parse_func = partial(parse_gebd_tf_record, target_fps=target_fps, target_size=target_size,
                         random_offset=(split == 'train'))  # (split == 'train')
    num_parallel_read = os.cpu_count() #// 2
    autotune = tf.data.AUTOTUNE
    if split == 'train':
        n_time = 32 * n_gpus
        prefetch = max(256, batch_size * n_time)
    else:
        n_time = 64 * n_gpus
        prefetch = max(256, batch_size * n_time)

    dataset = tf.data.TFRecordDataset(list_files, num_parallel_reads=num_parallel_read).prefetch(autotune)
    # for dm in dataset:
    #     x = parse_func(dm)
    #     pass
    dataset = dataset.map(parse_func, num_parallel_calls=autotune, deterministic=True).prefetch(prefetch)
    if split == 'train':
        dataset = dataset.shuffle(prefetch)

    dataset = dataset.batch(batch_size, num_parallel_calls=autotune, deterministic=True).prefetch(autotune)
    return dataset


if __name__ == '__main__':
    split = 'train'
    root_path = '/home/hvthong/sXProject/LoveU_TF/'
    data_root = '/mnt/SharedProject/Dataset/LOVEU_22/gebd/frames'
    # set_gpu_growth()
    img_size = 224

    # create_tf_records(root_path, data_root, target_fps=25, target_size=img_size, split='train', num_shards=100)
    # create_tf_records(root_path, data_root, target_fps=25, target_size=img_size, split='val')
    # create_tf_records(root_path, data_root, target_fps=25, target_size=img_size, split='test')
