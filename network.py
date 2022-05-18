import os
import sys

import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tf_debug
from parse import parse
from tqdm import trange
from ops import tf_matrix_vector_mul, tf_add_vectors, tf_matrix_from_quaternion, tf_sub_vectors
#from test import test_process, test_process_refine
from test import test_process
from data.data import prepareBatch,dataTransform
from data.my_data import myPrepareBatch

class Network(object):

    def  __init__(self, config):

        self.config = config

        gpu_flag = True

        # if config.gpu_options == 'gpu':
        #     device_name = '/' + config.gpu_options + ':' + config.gpu_number
        #     gpu_flag = True
        # else:
        #     device_name = '/' + config.gpu_options + ':0'

        self._init_tensorflow(gpu_flag)

        # with tf.device(device_name):
        self._build_placeholder()
        self._build_loss_func()
        self._build_model()
        self._build_loss()
        self._build_summary()
        self._build_optim()
        self._build_writer()



    def _init_tensorflow(self, gpu_flag):

        if not gpu_flag:
            #limit CPU threads with OMP_NUM_THREADS
            num_threads = (int)(os.popen('grep -c cores /proc/cpuinfo').read())
            if num_threads != "":
                num_threads = int(num_threads)
                print("limiting tensorflow to {} threads!".format(num_threads))
                # Limit
                tfconfig = tf.ConfigProto(
                    intra_op_parallelism_threads=num_threads,
                    inter_op_parallelism_threads=num_threads,
                )
            else:
                tfconfig = tf.ConfigProto()

        else:
            gpu_options = tf.GPUOptions(allow_growth=True)
            tfconfig = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True, gpu_options=gpu_options)

        self.sess = tf.Session(config=tfconfig)

    def _build_placeholder(self):


        self.x_in_1 = tf.placeholder(tf.float32, [None, 1, None, 3], name="x_in1")
        self.x_in_2 = tf.placeholder(tf.float32, [None, 1, None, 3], name="x_in2")
        self.x_in = tf.concat([self.x_in_1, self.x_in_2], 1)

        self.R_in = tf.placeholder(tf.float32, [None, 9], name="R_in")
        self.t_in = tf.placeholder(tf.float32, [None, 3], name="t_in")
        self.f_in = tf.placeholder(tf.float32, [None, None], name="f_in")
        self.is_training = tf.placeholder(tf.bool, (), name="is_training")

        # Global step for optimization
        self.global_step = tf.get_variable(
            "global_step", shape=(),
            initializer=tf.zeros_initializer(),
            dtype=tf.int64,
            trainable=False)

    def _build_loss_func(self):

        from ops import l1, l2, geman_mcclure, l05

        print('Loss Function Selected - {}'.format(self.config.loss_function))

        if self.config.loss_function == 'l1':
            self.loss_function = l1

        elif self.config.loss_function == 'l2' or self.config.loss_function == 'wls':
            self.loss_function = l2

        elif self.config.loss_function == 'gm':
            self.loss_function = geman_mcclure

        elif self.config.loss_function == 'l05':
            self.loss_function = l05

    def procrustes(self):
        x1_ = tf.squeeze(self.x_in_1, axis=1)
        sh = tf.shape(x1_)
        x2_ = tf.squeeze(self.x_in_2, axis=1)
        nor_x2 = tf_sub_vectors(x2_, self.t_hat)
        weights_ = tf.expand_dims(self.weights, 2)
        x1_ = weights_ * x1_
        nor_x2_t = tf.transpose(nor_x2, [0, 2, 1])
        M = tf.matmul(nor_x2_t, x1_)
        s, u, v = tf.linalg.svd(M)
        Det = tf.matrix_determinant(u) * tf.matrix_determinant(v)
        eye = tf.constant([[1, 0, 0], [0, 1, 0], [0, 0, 0]], float)
        tmp = tf.constant([[0, 0, 0], [0, 0, 0], [0, 0, 1]], float)
        eye = tf.tile(tf.expand_dims(eye, 0), [sh[0], 1, 1])
        tmp = tf.tile(tf.expand_dims(tmp, 0), [sh[0], 1, 1])
        Det = tf.expand_dims(tf.expand_dims(Det, 1), 2)
        Det = tf.tile(Det, [1, 3, 3])
        Det = tf.to_float(Det > 0)
        eye = eye + tmp * Det

        U_mul_s = tf.matmul(u, eye)
        R = tf.matmul(U_mul_s, tf.transpose(v, [0, 2, 1]))
        return R

    def rigid_transform_3d(A, B, weights=None, weight_threshold=0):
        """
        Input:
            - A:       [bs, num_corr, 3], source point cloud
            - B:       [bs, num_corr, 3], target point cloud
            - weights: [bs, num_corr]     weight for each correspondence
            - weight_threshold: float,    clips points with weight below threshold
        Output:
            - R, t
        """
        bs = A.shape[0]
        if weights is None:
            weights = torch.ones_like(A[:, :, 0])
        weights[weights < weight_threshold] = 0
        # weights = weights / (torch.sum(weights, dim=-1, keepdim=True) + 1e-6)

        # find mean of point cloud
        centroid_A = torch.sum(A * weights[:, :, None], dim=1, keepdim=True) / (
                    torch.sum(weights, dim=1, keepdim=True)[:, :, None] + 1e-6)
        centroid_B = torch.sum(B * weights[:, :, None], dim=1, keepdim=True) / (
                    torch.sum(weights, dim=1, keepdim=True)[:, :, None] + 1e-6)

        # subtract mean
        Am = A - centroid_A
        Bm = B - centroid_B

        # construct weight covariance matrix
        Weight = torch.diag_embed(weights)
        H = Am.permute(0, 2, 1) @ Weight @ Bm

        # find rotation
        U, S, Vt = torch.svd(H.cpu())
        U, S, Vt = U.to(weights.device), S.to(weights.device), Vt.to(weights.device)
        delta_UV = torch.det(Vt @ U.permute(0, 2, 1))
        eye = torch.eye(3)[None, :, :].repeat(bs, 1, 1).to(A.device)
        eye[:, -1, -1] = delta_UV
        R = Vt @ eye @ U.permute(0, 2, 1)
        t = centroid_B.permute(0, 2, 1) - R @ centroid_A.permute(0, 2, 1)
        # warp_A = transform(A, integrate_trans(R,t))
        # RMSE = torch.sum( (warp_A - B) ** 2, dim=-1).mean()
        return integrate_trans(R, t)

    def post_refinement(self, ):
        """
        Perform post refinement using the initial transformation matrix, only adopted during testing.
        Input
            - initial_trans: [bs, 4, 4]
            - src_keypts:    [bs, num_corr, 3]
            - tgt_keypts:    [bs, num_corr, 3]
            - weights:       [bs, num_corr]
        Output:
            - final_trans:   [bs, 4, 4]
        """
        R_hat_tmp = self.R_hat
        t_hat_tmp = self.t_hat

        x1_ = tf.squeeze(self.x_in_1, axis=1)
        sh = tf.shape(x1_)
        x2_ = tf.squeeze(self.x_in_2, axis=1)

        inlier_threshold = 0.10
        if inlier_threshold == 0.10:  # for 3DMatch
            inlier_threshold_list = [0.10] * 20
        else:  # for KITTI
            inlier_threshold_list = [1.2] * 20

        previous_inlier_num = 0
        for inlier_threshold in inlier_threshold_list:
            warped_src_keypts = tf_matrix_vector_mul(R_hat_tmp, x1_)
            warped_src_keypts = tf_add_vectors(warped_src_keypts, t_hat_tmp)
            L2_dis = tf.norm(warped_src_keypts - x2_, axis=-1)
            pred_inlier = (L2_dis < inlier_threshold) # assume bs = 1
            pred_inlier = tf.to_float(pred_inlier)

            weights = 1 / (1 + (L2_dis / inlier_threshold) ** 2)

            weights = weights * pred_inlier


            centroid_1 = tf.reduce_mean(x1_, axis=1)
            centroid_2 = tf.reduce_mean(x2_, axis=1)
            nor_x1 = tf_sub_vectors(x1_, centroid_1)
            nor_x2 = tf_sub_vectors(x2_, centroid_2)
            weights_ = tf.expand_dims(weights, 2)
            nor_x1 = weights_ * nor_x1
            nor_x2_t = tf.transpose(nor_x2, [0, 2, 1])
            M = tf.matmul(nor_x2_t, nor_x1)
            s, u, v = tf.linalg.svd(M)
            Det = tf.matrix_determinant(u) * tf.matrix_determinant(v)
            eye = tf.constant([[1, 0, 0], [0, 1, 0], [0, 0, 0]], float)
            tmp = tf.constant([[0, 0, 0], [0, 0, 0], [0, 0, 1]], float)
            eye = tf.tile(tf.expand_dims(eye, 0), [sh[0], 1, 1])
            tmp = tf.tile(tf.expand_dims(tmp, 0), [sh[0], 1, 1])
            Det = tf.expand_dims(tf.expand_dims(Det, 1), 2)
            Det = tf.tile(Det, [1, 3, 3])
            Det = tf.to_float(Det > 0)
            eye = eye + tmp * Det

            U_mul_s = tf.matmul(u, eye)
            R_hat_tmp = tf.matmul(U_mul_s, tf.transpose(v, [0, 2, 1]))


        return R_hat_tmp, t_hat_tmp


    def regress_R(self):

        from ops import tf_skew_symmetric

        if self.config.representation == 'lie':
            self.skew = tf_skew_symmetric(self.R_hat)
            self.R_hat = tf.reshape(self.skew, [-1, 3, 3])
            self.R_hat = tf.linalg.expm(self.R_hat)

        elif self.config.representation == 'quat':
            self.R_hat = tf_matrix_from_quaternion(self.R_hat)
            self.R_hat = tf.reshape(self.R_hat, [-1, 3, 3])

        elif self.config.representation == 'linear':
            self.R_hat = tf.reshape(self.R_hat, [-1, 3, 3])

        else:
            print('Not a valid representation')
            exit(10)



    def _build_model(self):

        # For determining the runtime shape
        # x_shp = tf.shape(self.x_in)

        # -------------------- Network architecture --------------------
        # Import correct build_graph function
        from archs.arch import build_drift_graph
        print("Building Graph")
        self.logits, self.weight, self.R_hat, self.t_hat, self.middle_t = build_drift_graph(self.x_in, self.is_training, self.config)
        # ---------------------------------------------------------------

        self.weights = tf.nn.sigmoid(self.weight)

        # self.weights = tf.nn.relu(tf.tanh(self.logits))

        if self.config.R_method == "svd":
            self.R_hat = self.procrustes()
        else:
            self.R_hat = self.regress_R()

        if self.config.post_refine == True:
            self.R_hat, self.t_hat = self.post_refinement()

    def _build_loss(self):
        """Build our cross entropy loss."""

        with tf.variable_scope("Loss", reuse=tf.AUTO_REUSE):

            # sh = tf.shape(self.x_in_2)
            x1_ = tf.squeeze(self.x_in_1, axis=1)
            x2_ = tf.squeeze(self.x_in_2, axis=1)


            print(self.R_hat.shape)
            self.x2_hat = tf_matrix_vector_mul(self.R_hat, x1_)
            self.x2_hat = tf_add_vectors(self.x2_hat, self.t_hat)
            sub = self.x2_hat - x2_

            if self.config.loss_function == 'wls':
                w_mul = tf.expand_dims(self.weights, axis=2)
                w_mul1 = tf.tile(w_mul, [1, 1, 3])
                sub = w_mul1*sub

            print(sub.shape)
            r_loss = self.loss_function(sub)
            # r_loss = tf.reduce_sum(r_loss, axis=1)
            r_loss = tf.reduce_sum(r_loss * self.f_in, axis=1) # filter the outliers: modified by yangfan


            t_loss = tf.reduce_mean(tf.reduce_sum(tf.abs(self.t_in - self.t_hat), 1))


            if self.config.drift_loss_type == 'alignment':
                drift_loss = 0
            else:
                drift_loss = tf.reduce_mean(tf.reduce_sum(
                    tf.abs(tf.expand_dims(self.t_in, axis=1) - self.middle_t), axis=-1
                ))


            self.t_loss = 2 * t_loss
            tf.summary.scalar("t_loss", self.t_loss)

            self.rec_loss = tf.reduce_mean(r_loss)
            self.rec_loss =(self.config.loss_reconstruction * self.rec_loss * tf.to_float(
                            self.global_step >= tf.to_int64(
                            self.config.loss_reconstruction_init_iter)))

            tf.summary.scalar("reconstruction_loss", self.rec_loss)

            is_pos = tf.to_float(self.f_in > 0)
            is_neg = tf.to_float(self.f_in <= 0)
            c = is_pos - is_neg

            # clss
            clf_losses = -tf.log(tf.nn.sigmoid(c * self.logits))

            num_pos = tf.nn.relu(tf.reduce_sum(is_pos, axis=1) - 1.0) + 1.0
            num_neg = tf.nn.relu(tf.reduce_sum(is_neg, axis=1) - 1.0) + 1.0
            classif_loss_p = tf.reduce_sum(
                clf_losses * is_pos, axis=1
            )
            classif_loss_n = tf.reduce_sum(
                clf_losses * is_neg, axis=1
            )
            self.clf_loss = tf.reduce_mean(
                classif_loss_p * 0.5 / num_pos +
                classif_loss_n * 0.5 / num_neg
            )

            self.clf_loss = self.config.loss_classif * self.clf_loss
            tf.summary.scalar("classification_loss", self.clf_loss)
            tf.summary.scalar(
                "classif_loss_p",
                tf.reduce_mean(classif_loss_p * 0.5 / num_pos))
            tf.summary.scalar(
                "classif_loss_n",
                tf.reduce_mean(classif_loss_n * 0.5 / num_neg))

            # L2 loss

            for var in tf.trainable_variables():
                if "weights" in var.name:
                    tf.add_to_collection("l2_losses", tf.reduce_sum(var ** 2))
            l2_loss = tf.add_n(tf.get_collection("l2_losses"))
            tf.summary.scalar("l2_loss", l2_loss)

            self.loss = self.config.loss_decay * l2_loss

            self.loss += self.rec_loss
            self.loss += self.clf_loss
            self.loss += self.t_loss

            tf.summary.scalar("loss", self.loss)

    def _build_summary(self):
        """Build summary ops."""

        # Merge all summary op
        self.summary_op = tf.summary.merge_all()

    def _build_optim(self):
        """Build optimizer related ops and vars."""

        with tf.variable_scope("Optimization", reuse=tf.AUTO_REUSE):
            learning_rate = self.config.train_lr
            max_grad_norm = None
            optim = tf.train.AdamOptimizer(learning_rate)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                grads_and_vars = optim.compute_gradients(self.loss)
                self.grads = grads_and_vars
                # gradient clipping
                if max_grad_norm is not None:
                    new_grads_and_vars = []
                    for idx, (grad, var) in enumerate(grads_and_vars):
                        if grad is not None:
                            new_grads_and_vars.append((
                                tf.clip_by_norm(grad, max_grad_norm), var))
                    grads_and_vars = new_grads_and_vars

                # Check numerics and report if something is going on. This
                # will make the backward pass stop and skip the batch
                new_grads_and_vars = []
                for idx, (grad, var) in enumerate(grads_and_vars):
                    if grad is not None:
                        grad = tf.check_numerics(
                            grad, "Numerical error in gradient for {}"
                            "".format(var.name))
                    new_grads_and_vars.append((grad, var))


                # Should only apply grads once they are safe
                self.optim = optim.apply_gradients(
                    new_grads_and_vars, global_step=self.global_step)

            # Summarize all gradients
            for grad, var in grads_and_vars:
                if grad is not None:
                    tf.summary.histogram(var.name + '/gradient', grad)

    # TODO Mudar os writers (por o graph)
    def _build_writer(self):
        """Build the writers and savers"""

        # Create suffix automatically if not provided
        suffix_tr = self.config.log_dir
        if suffix_tr == "":
            suffix_tr = "-".join(sys.argv)
        suffix_te = self.config.test_log_dir
        if suffix_te == "":
            suffix_te = suffix_tr

        # Directories for train/test
        self.res_dir_tr = os.path.join(self.config.res_dir, suffix_tr)
        self.res_dir_va = os.path.join(self.config.res_dir, suffix_te)
        self.res_dir_te = os.path.join(self.config.res_dir, suffix_te)

        # Create summary writers
        if self.config.run_mode == "train":
            self.summary_tr = tf.summary.FileWriter(
                os.path.join(self.res_dir_tr, "train", "logs"))
            self.summary_tr.add_graph(tf.get_default_graph())
        if self.config.run_mode != "valid":
            self.summary_va = tf.summary.FileWriter(
                os.path.join(self.res_dir_va, "valid", "logs"))
        if self.config.run_mode == "test":
            self.summary_te = tf.summary.FileWriter(
                os.path.join(self.res_dir_te, "test", "logs"))

        # Create savers (one for current, one for best)
        self.saver_cur = tf.train.Saver()
        self.saver_best = tf.train.Saver()
        # Save file for the current model
        self.save_file_cur = os.path.join(
            self.res_dir_tr, "model")
        # Save file for the best model
        self.save_file_best = os.path.join(
            self.res_dir_tr, "model-cur_best")

        # Other savers
        self.mean_frob_file = os.path.join(self.res_dir_va, "valid", "mean_frob.txt")

    def train(self, data):

        best_mean_frob = 101
        latest_checkpoint = tf.train.latest_checkpoint(self.res_dir_tr)
        # latest_checkpoint = './logs/3d_match_retrain2/model-3762'
        b_resume = latest_checkpoint is not None
        if b_resume:
            # Restore network
            print("Restoring from {}...".format(self.res_dir_tr))
            self.saver_cur.restore(self.sess, latest_checkpoint)

            # restore number of steps so far
            init_epoch = self.sess.run(self.global_step % len(data['train']['x1']))
            # restore best validation result
            if os.path.exists(self.mean_frob_file):
                with open(self.mean_frob_file, "r") as ifp:
                    dump_res = ifp.read()
                dump_res = parse("{best_mean_frob:e}\n", dump_res)
                best_mean_frob = dump_res["best_mean_frob"]
        else:

            if self.config.data_aug:
                suffix_tr = self.config.aug_dir
                self.res_dir_restore = os.path.join(self.config.res_dir, suffix_tr)
                self.res_dir_restore = os.path.join(self.res_dir_restore, "model-cur_best")

                # Restore network
                print("Data Augmentation from {}...".format(self.res_dir_restore))
                # self.saver_cur.restore(self.sess, self.res_dir_restore) commended by yangfan

            print("Starting from scratch...")
            init_epoch = 0

        # ----------------------------------------
        # The training loop
        print("Initializing...")
        self.sess.run(tf.global_variables_initializer())

        self.saver_cur.save(self.sess, self.save_file_cur, global_step=self.global_step, write_meta_graph=True)

        batch_size = self.config.train_batch_size
        max_epoch = self.config.train_epoch
        max_iter = self.config.train_step
        step = 0

        for epoch in trange(init_epoch, max_epoch, ncols=self.config.tqdm_width):

            x1_tr, x2_tr, Rs_tr, ts_tr, fs_tr, max_iter = myPrepareBatch(data['train'], batch_size)

            if self.config.data_aug:

                if self.config.aug_cl:
                    x1_tr, Rs_tr, ts_tr = dataTransform(x1_tr, Rs_tr, ts_tr, epoch, aug_cl=True)
                else:
                    x1_tr, Rs_tr, ts_tr = dataTransform(x1_tr, Rs_tr, ts_tr, step)

            for idx in trange(0, max_iter - 1, ncols=self.config.tqdm_width):
                # ----------------------------------------
                # Batch construction

                numkps = np.array([x1_tr[idx][_i].shape[0] for _i in range(len(x1_tr[idx]))])
                cur_num_kp = numkps.min()

                # Actual construction of the batch
                x1_b = np.array([x1_tr[idx][_i][:cur_num_kp, :] for _i in range(len(x1_tr[idx]))]
                                ).reshape((len(x1_tr[idx]), 1, cur_num_kp, 3))

                x2_b = np.array([x2_tr[idx][_i][:cur_num_kp, :] for _i in range(len(x1_tr[idx]))]
                                ).reshape((len(x1_tr[idx]), 1, cur_num_kp, 3))

                Rs_b = np.array([Rs_tr[idx][_i] for _i in range(len(x1_tr[idx]))]).reshape(len(x1_tr[idx]), 9)
                ts_b = np.array([ts_tr[idx][_i] for _i in range(len(x1_tr[idx]))]).reshape(len(x1_tr[idx]), 3)
                fs_b = np.array([fs_tr[idx][_i][:cur_num_kp] for _i in range(len(x1_tr[idx]))]).reshape(len(x1_tr[idx]),
                                                                                                        cur_num_kp)

                # ----------------------------------------
                # Train

                # Feed Dict
                feed_dict = {
                    self.x_in_1: x1_b,
                    self.x_in_2: x2_b,
                    self.R_in: Rs_b,
                    self.t_in: ts_b,
                    self.f_in: fs_b,
                    self.is_training: True,
                }

                b_write_summary = ((step + 1) % self.config.report_intv) == 0
                b_validate = (idx - (max_iter - 2)) == 0

                if b_validate:
                    fetch = {"summary": self.summary_op, "global_step": self.global_step}
                elif b_write_summary:
                    fetch = {"optim": self.optim, "rec_loss": self.rec_loss, "summary": self.summary_op,
                             "global_step": self.global_step}
                else:
                    fetch = {"optim": self.optim, "rec_loss": self.rec_loss}

                try:
                    res = self.sess.run(fetch, feed_dict=feed_dict)



                except tf.errors.InvalidArgumentError:
                    print("Backward pass had numerical errors.")
                    continue

                    # Write summary and save current model
                if b_write_summary:
                    self.summary_tr.add_summary(
                        res["summary"], global_step=res["global_step"])
                    self.saver_cur.save(self.sess, self.save_file_cur, global_step=self.global_step,
                                        write_meta_graph=False)

                step += 1

            b_validate = True
            if b_validate:
                # valid = {'x1': x1_tr[idx+1], 'x2': x2_tr[idx+1], 'R': Rs_tr[idx+1], 't': ts_tr[idx+1], 'flag': fs_tr[idx+1]}

                valid = data['train']
                cur_global_step = res["global_step"]
                mean_frob = test_process('valid', self.sess, valid, cur_global_step, self.summary_va,
                                         self.config,
                                         self.x_in_1, self.x_in_2, self.R_in, self.t_in, self.f_in,
                                         self.is_training,
                                         self.R_hat, self.t_hat, self.logits, self.rec_loss, self.weights,
                                         self.config.reg_flag, self.config.reg_function)

                if mean_frob < best_mean_frob and step > self.config.loss_reconstruction_init_iter:
                    print(
                        "Saving best model with mean_frob = {}".format(
                            mean_frob))
                    best_mean_frob = mean_frob
                    # Save best validation result
                    with open(self.mean_frob_file, "w") as ofp:
                        ofp.write("{:e}\n".format(best_mean_frob))
                    # Save best model
                    self.saver_best.save(self.sess, self.save_file_best, write_meta_graph=False)


    def test(self, data):

        # Check if model exists
        if not os.path.exists(self.save_file_best + ".index"):
            print("Model File {} does not exist! Quiting".format(
                self.save_file_best))
            exit(1)

        # Restore model
        self.saver_best.restore(
            self.sess,
            self.save_file_best)

        print("Restoring from {}...".format(
            self.save_file_best))

        # Run Test
        cur_global_step = 0 # dummy


        test_process('test', self.sess, data['test'], cur_global_step, self.summary_te, self.config,
                     self.x_in_1, self.x_in_2, self.R_in, self.t_in, self.f_in, self.is_training,
                     self.R_hat, self.t_hat, self.logits, self.rec_loss, self.weights,
                     self.config.reg_flag, self.config.reg_function)


        return 0
