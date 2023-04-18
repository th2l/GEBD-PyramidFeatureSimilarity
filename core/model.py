"""
Author: Van Thong Huynh
Affiliation: Dept. of AI Convergence, Chonnam Nat'l Univ.
"""
import sys

import tensorflow as tf
import keras
import keras.layers as layers

from core.backbone import ResNetBackbone
import tensorflow_io as tfio
from keras import mixed_precision
from keras.mixed_precision import loss_scale_optimizer as lso

import tensorflow_models as tfm


def get_filtered_output(inputs, size=5):
    x_filter = tf.expand_dims(inputs, axis=1)  # Convert to N x 1 x seq_len x 1
    x_filter = tfio.experimental.filter.gaussian(x_filter, ksize=(1, size), sigma=1., mode='SYMMETRIC')
    x_filter = tf.squeeze(x_filter, axis=1)
    return x_filter


def get_distance(seq_len, input_feature, rate=1):
    # print(tf.shape(input_feature))

    l2norm_input = tf.cast(input_feature, dtype=tf.float32)
    l2norm_input = tf.nn.l2_normalize(l2norm_input, axis=-1)

    feature_padded = tf.pad(l2norm_input, paddings=([0, 0], [rate, rate], [0, 0]))

    ret = []
    for rt in range(rate):
        left_dist = tf.reduce_sum(
            tf.math.squared_difference(l2norm_input, feature_padded[:, rt:rt + seq_len, :]),
            axis=-1,
            keepdims=True)
        right_dist = tf.reduce_sum(
            tf.math.squared_difference(l2norm_input, feature_padded[:, rate + rt:rate + rt + seq_len, :]),
            axis=-1,
            keepdims=True)

        ret.append(left_dist)
        ret.append(right_dist)

    ret = tf.concat(ret, axis=-1)

    if mixed_precision.global_policy().name == 'mixed_float16':
        ret = tf.cast(ret, dtype=tf.float16)

    return ret


class TemporalPyramidPooling(layers.Layer):
    # Based on https://github.com/tensorflow/models/blob/master/official/vision/modeling/layers/deeplab.py#L22
    # Latest commit 65174bd
    def __init__(self, output_channels, dilation_rates, output_dim=64, kernel_initializer='HeNormal',
                 activation='relu', tpp_feat=False, dropout=0., **kwargs):
        super(TemporalPyramidPooling, self).__init__(**kwargs)
        self.output_channels = output_channels
        self.dilation_rates = dilation_rates
        # self.pool_kernel_size = pool_kernel_size
        self.kernel_initializer = kernel_initializer
        self.tpp_feat = tpp_feat
        self.dropout = dropout
        self.output_dim = output_dim
        self.activation = activation

    def create_block(self, n_filters=-1, kernel_size=3, dilation_rate=1, epsilon=1e-3,
                     activation='relu', name=None, incl_gaussian_noise=False):
        if n_filters < 0:
            n_filters = self.output_channels

        block_layers = [
            layers.Conv1D(filters=n_filters, kernel_size=kernel_size, dilation_rate=dilation_rate,
                          kernel_initializer=self.kernel_initializer, padding='same'),
            layers.LayerNormalization(epsilon=epsilon, dtype=tf.float32),
            layers.Activation(activation),
        ]
        if dilation_rate > 1:
            block_layers = [layers.DepthwiseConv1D(kernel_size=kernel_size, padding='same',
                                                   kernel_initializer=self.kernel_initializer), ] + block_layers

        # if incl_gaussian_noise:
        #     block_layers = block_layers + [layers.GaussianNoise(stddev=0.05), ]

        block = keras.Sequential(block_layers, name=name)
        return block

    def build(self, input_shape):
        self.seq_len = input_shape[1]
        n_features = input_shape[2]

        self.atpp_layers = []

        residual = False
        kn_size = 3
        activation = self.activation

        conv_sequential = self.create_block(kernel_size=kn_size, dilation_rate=1, name='D_1',
                                            activation=activation)

        self.atpp_layers.append(conv_sequential)

        for dilation_rate in self.dilation_rates:
            conv_sequential = self.create_block(kernel_size=kn_size, dilation_rate=dilation_rate,
                                                name=f'D_{dilation_rate}', activation=activation)

            self.atpp_layers.append(conv_sequential)

        self.projection = self.create_block(n_filters=self.output_dim, kernel_size=1, dilation_rate=1,
                                            name=f'{self.name}_projection', activation=activation)
        self.dropout_layer = layers.Dropout(self.dropout)

        if self.tpp_feat and len(self.dilation_rates) > 0:
            self.projection_feat = self.create_block(n_filters=self.output_channels, kernel_size=1, dilation_rate=1,
                                                     name=f'{self.name}_projection_feat',
                                                     activation=activation)
            # self.dropout_feat = layers.Dropout(self.dropout)

    def call(self, inputs, training=None):
        if training is None:
            training = tf.keras.backend.learning_phase()

        result = []
        result_feat = []

        for layer in self.atpp_layers:
            feat_input = layer(inputs, training=training)
            if self.tpp_feat:
                result_feat.append(feat_input)
            input_feature = feat_input + inputs
            feat_distance = get_distance(seq_len=self.seq_len, input_feature=input_feature, rate=self.seq_len // 10)
            result.append(feat_distance)

        if self.tpp_feat and len(self.dilation_rates) > 0:
            result_feat = tf.concat(result_feat, axis=-1)
            result_feat = self.projection_feat(result_feat, training=training)
            result_feat = get_distance(seq_len=self.seq_len, input_feature=result_feat, rate=self.seq_len // 10)
            # result_feat = self.dropout_feat(result_feat, training=training)
            result.append(result_feat)
        else:
            result_feat = None

        result = tf.concat(result, axis=-1)

        result = self.projection(result, training=training)
        result = self.dropout_layer(result, training=training)

        return result

    def get_config(self):
        config_dict = {'output_channels': self.output_channels, 'dilation_rates': self.dilation_rates,
                       'dropout': self.dropout, 'tpp_feat': self.tpp_feat, 'output_dim': self.output_dim,
                       'kernel_initializer': self.kernel_initializer, 'activation': self.activation}
        return config_dict


class LinearFusion(layers.Layer):
    def __init__(self, n_input=1, name=None, **kwargs):
        super(LinearFusion, self).__init__(name=name, **kwargs)
        self._n_input = n_input
        self._config_dict = {'n_input': n_input}

    def build(self, input_shape):
        if self._n_input > 1:
            self.linear_weight = self.add_weight("linear_weight", shape=(1, 1, self._n_input), dtype=tf.float32,
                                                 initializer=tf.keras.initializers.Zeros(), trainable=True)
        else:
            self.linear_weight = None

    def call(self, inputs):
        if self._n_input > 1:
            fuse_weight = tf.nn.softmax(self.linear_weight)
            out = tf.reduce_sum(fuse_weight * inputs, axis=-1)
            return out
        else:
            return inputs

    def get_config(self):
        return self._config_dict


class PredictionHead(layers.Layer):
    def __init__(self, dim_ff=512, dropout=0., kernel_initializer='HeNormal', activation='relu', name=None, **kwargs):
        super(PredictionHead, self).__init__(name=name, **kwargs)
        self._dim_ff = dim_ff
        self._kernel_initializer = kernel_initializer
        self._dropout = dropout
        self._activation = activation
        self._config_dict = {'dim_ff': dim_ff, 'kernel_initializer': kernel_initializer, 'dropout': dropout,
                             'name': name, 'activation': activation}

    def build(self, input_shape):
        self.head_sequential = keras.Sequential([
            layers.Conv1D(self._dim_ff, kernel_size=3, padding='same', kernel_initializer='HeNormal'),
            layers.Activation(self._activation),
            layers.Dropout(self._dropout),
            layers.Conv1D(1, kernel_size=1, padding='same', kernel_initializer='HeNormal'),
            # layers.GaussianNoise(stddev=0.1)  # stddev=0.1 => 0.77039 https://wandb.ai/hvthong298/Kinetics-GEBD_v2/runs/20230214_004945
        ])

    def call(self, inputs, training=None):
        if training is None:
            training = tf.keras.backend.learning_phase()

        outs = inputs
        # outs = tf.cast(tf.reshape(inputs, (-1, 256)), tf.float32)
        # outs = self.rff_layer(outs)
        # outs = tf.cast(tf.reshape(outs, (-1, 50, self._dim_ff)), tf.float16)

        return self.head_sequential(outs, training=training)

    def get_config(self):
        return self._config_dict


class GebProjection(layers.Layer):
    def __init__(self, output_projection, activation, sd_dilation_rates, return_logits=0, name=None, **kwargs):
        super(GebProjection, self).__init__(name=name, **kwargs)

        self._output_projection = output_projection
        self._activation_projection = activation
        self._dilation_rates = [2, 4, 8]
        self._sd_dilation_rates = sd_dilation_rates
        self._return_logits = return_logits
        self._config_dict = {'output_projection': output_projection, 'activation': activation,
                             'return_logits': return_logits,
                             'sd_dilation_rates': sd_dilation_rates, 'name': name}

    def build(self, input_shape):

        if self._return_logits > 1:
            self.logits_layer = keras.Sequential(
                [layers.Conv1D(self._output_projection, kernel_size=3, padding='same', kernel_initializer='HeNormal'),
                 layers.Activation(activation=self._activation_projection),
                 layers.Conv1D(self._return_logits, kernel_size=1, padding='same', kernel_initializer='HeNormal'),
                 layers.Activation(activation='sigmoid', dtype=tf.float32)
                 ])

        else:
            self.projection = keras.Sequential(
                [layers.Conv1D(self._output_projection, kernel_size=1, kernel_initializer='HeNormal'),
                 layers.LayerNormalization(dtype=tf.float32),
                 layers.Activation(activation=self._activation_projection)])

            # Temporal projection
            temporal_proj = []
            if self._sd_dilation_rates <= 0:
                sd_dilation_rates = [2, 4, 8]  # dilation_rates
            else:
                sd_dilation_rates = [self._sd_dilation_rates] * len(self._dilation_rates)

            for dilation in sd_dilation_rates:
                print('Dilation = ', dilation)
                cur_block = [
                    layers.Conv1D(self._output_projection, kernel_size=3, dilation_rate=dilation, padding='same',
                                  kernel_initializer='HeNormal'),
                    layers.LayerNormalization(dtype=tf.float32),
                    layers.Activation(activation=self._activation_projection)]

                temporal_proj += cur_block

            self.temporal_proj = keras.Sequential(temporal_proj)

    def call(self, inputs, training=None):
        if training is None:
            training = tf.keras.backend.learning_phase()

        if self._return_logits > 1:
            outs = inputs
            outs = self.logits_layer(outs, training=training)
            outs = tf.expand_dims(outs, axis=2)

            outs = tf.cast(outs, inputs.dtype)
        else:
            outs = self.projection(inputs, training=training)
            outs = self.temporal_proj(outs, training=training)
        return outs

    def get_config(self):
        return self._config_dict


class ProbGebd(tf.keras.Model):
    def __init__(self, seq_len=50, from_logits=True, level=0, prediction_dim=512, tpp_feat=False, output_dim=1024,
                 sd_dilation_rates=-1, **kwargs):

        inputs = layers.Input(shape=(seq_len, 224, 224, 3), name='image')
        endpoints = ResNetBackbone(weights='imagenet', level=level)(tf.reshape(inputs, (-1, 224, 224, 3)))

        num_feats = {'feat_last': 2048, 'feat_conv2': 256, 'feat_conv3': 512, 'feat_conv4': 1024}

        activation = 'gelu'
        dilation_rates = [2, 4, 8]  # [2, 4] #  # (2, 4, 8)
        dropout = 0.3

        out_list = []
        # out_feat = []
        # out_dist = []

        for ky in endpoints.keys():
            cur_inp = tf.reshape(endpoints[ky], (-1, seq_len, num_feats[ky]))
            cur_out = TemporalPyramidPooling(output_channels=num_feats[ky], dilation_rates=dilation_rates,
                                             output_dim=output_dim, tpp_feat=tpp_feat, dropout=dropout,
                                             activation=activation, name=f'tpp_{ky}')(cur_inp)
            # cur_out = keras.Sequential([layers.Conv1D(filters=output_dim, kernel_size=3, padding='same', kernel_initializer='HeNormal'),
            #                             layers.Activation(activation)])(cur_inp)
            out_list.append(cur_out)
            # out_feat.append(cur_out_feat)

        outs_dec = []
        dim_ff = prediction_dim
        out_dict = dict()

        if len(out_list) > 1:
            # for ky_idx in range(len(out_list)):
            #     cur_proj = GebProjection(output_dim, activation, sd_dilation_rates)(out_list[ky_idx])
            #     out_dict[f'outs_dec_{ky_idx}'] = PredictionHead(dim_ff=dim_ff, dropout=dropout, activation=activation)(
            #         cur_proj)
            outs_dec = layers.Concatenate()(out_list)
            # outs_dec = TemporalPyramidPooling(output_channels=output_dim*len(out_list), dilation_rates=dilation_rates,
            #                                   output_dim=output_dim, tpp_feat=tpp_feat, dropout=dropout,
            #                                   activation=activation)(outs_dec)
        else:
            outs_dec = outs_dec[0]

        outs_dec = GebProjection(output_dim, activation, sd_dilation_rates)(outs_dec)

        outs_dec = PredictionHead(dim_ff=dim_ff, dropout=dropout, activation=activation)(outs_dec)  # Main part

        out_dict['outs_dec'] = outs_dec

        # if tpp_feat:
        #     out_feat = PredictionHead(dim_ff=dim_ff * 2, dropout=dropout)(out_feat)
        #     out_dict.update({'out_feat': out_feat})

        if not from_logits:
            for ky in out_dict:
                out_dict[ky] = layers.Activation('sigmoid', dtype=tf.float32)(out_dict[ky])
                out_dict[ky] = get_filtered_output(out_dict[ky], size=seq_len // 10)

        else:
            for ky in out_dict:
                out_dict[ky] = tf.cast(out_dict[ky], dtype=tf.float32)

        list_out_keys = list(out_dict.keys())
        if len(list_out_keys) > 1:
            outs_final = tf.stack(list(out_dict.values()), axis=-1)
            outs_final = LinearFusion(n_input=len(list_out_keys), dtype=tf.float32)(outs_final)

            # out_dict.update({'outs': outs_final})

            out_dict = {'outs': outs_final}
        else:
            out_dict = {'outs': out_dict[list_out_keys[0]]}

        super(ProbGebd, self).__init__(inputs=[inputs, ], outputs=out_dict, **kwargs)

        self.from_logits = from_logits

        self._config_dict = {'seq_len': seq_len, 'from_logits': from_logits, 'level': level,
                             'prediction_dim': prediction_dim, 'output_dim': output_dim, 'tpp_feat': tpp_feat}

    def get_outputs_call(self, video_inputs, training=None):
        inputs_dict = {'image': video_inputs['image']}
        return self(inputs_dict, training=training)

    def train_step(self, data):
        if len(data) == 3:
            x, y, sample_weight = data
        else:
            sample_weight = None
            x, y = data

        loss_scale_opt = lso.BaseLossScaleOptimizer
        with tf.GradientTape() as tape:
            y_pred = self.get_outputs_call(x, training=True)

            loss = self.compiled_loss(y, y_pred['outs'], sample_weight=sample_weight,
                                      regularization_losses=self.losses)

            if isinstance(self.optimizer, loss_scale_opt):
                loss = self.optimizer.get_scaled_loss(loss)

        # Compute gradient
        tvars = self.trainable_variables

        if isinstance(self.optimizer, loss_scale_opt):
            grads = tape.gradient(loss, tvars)
            grads = self.optimizer.get_unscaled_gradients(grads)
        else:
            grads = tape.gradient(loss, tvars)

        self.optimizer.apply_gradients(list(zip(grads, tvars)))

        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        if len(data) == 3:
            x, y, sample_weight = data
        else:
            sample_weight = None
            x, y = data

        y_pred = self.get_outputs_call(x, training=False)

        if self.from_logits:
            y_pred = tf.nn.sigmoid(y_pred['outs'])
            y_pred = get_filtered_output(y_pred)
        else:
            y_pred = y_pred['outs']

        # Updates the metrics tracking the loss
        self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}

    def predict_step(self, data):
        if len(data) == 3:
            x, y, sample_weight = data
        else:
            sample_weight = None
            x, y = data

        y_pred = self.get_outputs_call(x, training=False)

        if self.from_logits:
            y_pred = tf.nn.sigmoid(y_pred['outs'])
            y_pred = get_filtered_output(y_pred)
        else:
            y_pred = y_pred['outs']

        return y_pred, x['vid_id'], x['original_dur']

    def get_config(self):
        return self._config_dict

    def _get_optimizer(self, optimizer):
        """Wraps `optimizer` in `LossScaleOptimizer` if necessary.
        https://github.com/keras-team/keras/issues/16560#issuecomment-1132276275
        """

        def _get_single_optimizer(opt):
            opt = tf.keras.optimizers.get(opt)
            if self.dtype_policy.name == "mixed_float16" and not isinstance(
                    opt, lso.BaseLossScaleOptimizer
            ):
                # Loss scaling is necessary with mixed_float16 for models to
                # converge to the same accuracy as with float32.
                opt = lso.BaseLossScaleOptimizer(opt)
            return opt

        return tf.nest.map_structure(_get_single_optimizer, optimizer)


class LossScaleOptimizerV31(lso.LossScaleOptimizerV3):
    def __init__(self, inner_optimizer, dynamic=True, initial_scale=None,
                 dynamic_growth_steps=None):
        super(LossScaleOptimizerV31, self).__init__(inner_optimizer, dynamic, initial_scale,
                                                    dynamic_growth_steps)

    @property
    def use_ema(self):
        return self._optimizer.use_ema

    @use_ema.setter
    def use_ema(self, use_ema):
        self._optimizer.use_ema = use_ema

    @property
    def ema_momentum(self):
        return self._optimizer.ema_momentum

    @ema_momentum.setter
    def ema_momentum(self, ema_momentum):
        self._optimizer.ema_momentum = ema_momentum
