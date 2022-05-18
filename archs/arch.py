import tensorflow as tf

from ops import conv1d_layer, CFD_block, globalmax_pool1d, regression_layer, regression_middle_layer, bn_act3, SCA_block



def build_drift_graph(x_in, is_training, config):

    activation_fn = tf.nn.relu

    x_in_shp = tf.shape(x_in)

    cur_input = x_in
    print(cur_input.shape)
    idx_layer = 0
    ksize = 1
    num_PCFD_layer = config.PCFD_layer
    num_SCA_layer = config.SCA_layer
    nchannel = config.net_nchannel
    act_pos = config.net_act_pos

    # First convolution
    with tf.variable_scope("hidden-input"):
        cur_input, _ = conv1d_layer(
            inputs=cur_input,
            ksize=1,
            nchannel=nchannel,
            activation_fn=None,
            perform_bn=False,
            perform_gcn=False,
            is_training=is_training,
            act_pos="pre",
            data_format="NHWC",
        )
        print(cur_input.shape)

    for _ksize, _nchannel in zip(
            [ksize] * num_PCFD_layer, [nchannel] * num_PCFD_layer):
        scope_name = "hidden-" + str(idx_layer)
        with tf.variable_scope(scope_name):
            dif, global_feature_offset, cur_input = CFD_block(
                inputs=cur_input,
                ksize=_ksize,
                nchannel=_nchannel,
                activation_fn=activation_fn,
                is_training=is_training,
                perform_bn=config.net_batchnorm,
                perform_gcn=config.net_gcnorm,
                act_pos=act_pos,
                data_format="NHWC",
            )


        if idx_layer == 0:
            global_feature_offsets = global_feature_offset
            feature_difs = dif
        else:
            global_feature_offsets = tf.concat([global_feature_offsets, global_feature_offset], axis=1)
            feature_difs = tf.concat([feature_difs, dif], axis=1)
        idx_layer += 1

    corr_feature = tf.nn.max_pool(feature_difs, [1, num_PCFD_layer, 1, 1], [1, 1, 1, 1], padding ='VALID')

    with tf.variable_scope("regression"):
    # with tf.variable_scope("output"):
        R_hat, t_hat = regression_layer(global_feature_offsets,
                                 nb_fc=256)
        middle_t = regression_middle_layer(global_feature_offsets)

    print(corr_feature.shape)
    for _ksize, _nchannel in zip(
            [ksize] * num_SCA_layer, [nchannel] * num_SCA_layer):
        scope_name = "hidden-" + str(idx_layer)
        with tf.variable_scope(scope_name):
            corr_feature = SCA_block(
                inputs=corr_feature,
                ksize=_ksize,
                nchannel=_nchannel,
                activation_fn=activation_fn,
                is_training=is_training,
                perform_bn=config.net_batchnorm,
                perform_gcn=config.net_gcnorm,
                act_pos=act_pos,
                data_format="NHWC",
            )
        idx_layer += 1

    with tf.variable_scope("output"):
        logits, _ = conv1d_layer(
            inputs=corr_feature,
            ksize=1,
            nchannel=1,
            activation_fn=None,
            is_training=is_training,
            perform_bn=False,
            perform_gcn=False,
            data_format="NHWC",
        )
        #  Flatten
        logits = tf.reshape(logits, (x_in_shp[0], x_in_shp[2]))

    with tf.variable_scope("weight"):
        weight, _ = conv1d_layer(
            inputs=corr_feature,
            ksize=1,
            nchannel=1,
            activation_fn=None,
            is_training=is_training,
            perform_bn=False,
            perform_gcn=False,
            data_format="NHWC",
        )
        #  Flatten
        weight = tf.reshape(weight, (x_in_shp[0], x_in_shp[2]))

    return logits, weight, R_hat, t_hat, middle_t


#
# arch.py ends here
