import datetime
import numpy as np
import time
from transformations import rotation_from_matrix
import sklearn.metrics as metrics
import pickle

import os
from registration.registration import initializePointCoud, globalRegistration, selectFunction, refine_registration, makeMatchesSet

import math
# from registration import preProcessData, globalRegistration, selectFunction, refine_registration

def saveToFile(dict):

    f = open('results/test.pickle', 'wb')
    pickle.dump(dict, f)
    f.close()



def RotationError(R1, R2):

    R1 = np.real(R1)
    R2 = np.real(R2)

    R_ = np.matmul(R1, np.linalg.inv(R2))
    ae = np.arccos(np.clip(((np.trace(R_) - 1) / 2), -1, 1))

    ae = np.rad2deg(ae)
    

    frob_norm = np.linalg.norm(R_ - np.eye(3), ord='fro')

    return ae, frob_norm

def TranslationError(t1, t2):

    return np.linalg.norm(t1-t2)



def integrate_trans(R, t):
    """
    Integrate SE3 transformations from R and t, support torch.Tensor and np.ndarry.
    Input
        - R: [3, 3] or [bs, 3, 3], rotation matrix
        - t: [3, 1] or [bs, 3, 1], translation matrix
    Output
        - trans: [4, 4] or [bs, 4, 4], SE3 transformation matrix
    """
    if len(R.shape) == 3:
        if isinstance(R, torch.Tensor):
            trans = torch.eye(4)[None].repeat(R.shape[0], 1, 1).to(R.device)
        else:
            trans = np.eye(4)[None]
        trans[:, :3, :3] = R
        trans[:, :3, 3:4] = t.view([-1, 3, 1])
    else:
        if isinstance(R, torch.Tensor):
            trans = torch.eye(4).to(R.device)
        else:
            trans = np.eye(4)
        trans[:3, :3] = R
        trans[:3, 3:4] = t
    return trans


def rigid_transform_3d(A, B, t, weights=None, weight_threshold=0):
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
    centroid_A = torch.sum(A * weights[:, :, None], dim=1, keepdim=True) / (torch.sum(weights, dim=1, keepdim=True)[:, :, None] + 1e-6)
    centroid_B = torch.sum(B * weights[:, :, None], dim=1, keepdim=True) / (torch.sum(weights, dim=1, keepdim=True)[:, :, None] + 1e-6)

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
    # t = centroid_B.permute(0,2,1) - R @ centroid_A.permute(0,2,1)
    # warp_A = transform(A, integrate_trans(R,t))
    # RMSE = torch.sum( (warp_A - B) ** 2, dim=-1).mean()
    return integrate_trans(R, t)

def transform(pts, trans):
    """
    Applies the SE3 transformations, support torch.Tensor and np.ndarry.  Equation: trans_pts = R @ pts + t
    Input
        - pts: [num_pts, 3] or [bs, num_pts, 3], pts to be transformed
        - trans: [4, 4] or [bs, 4, 4], SE3 transformation matrix
    Output
        - pts: [num_pts, 3] or [bs, num_pts, 3] transformed pts
    """
    if len(pts.shape) == 3:
        trans_pts = trans[:, :3, :3] @ pts.permute(0,2,1) + trans[:, :3, 3:4]
        return trans_pts.permute(0,2,1)
    else:
        trans_pts = trans[:3, :3] @ pts.T + trans[:3, 3:4]
        return trans_pts.T

def post_refinement(initial_trans, src_keypts, tgt_keypts, weights=None):
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
    assert initial_trans.shape[0] == 1
    inlier_threshold = 0.10
    if inlier_threshold == 0.10:  # for 3DMatch
        inlier_threshold_list = [0.10] * 10
    else:  # for KITTI
        inlier_threshold_list = [1.2] * 20

    previous_inlier_num = 0
    for inlier_threshold in inlier_threshold_list:
        warped_src_keypts = transform(src_keypts, initial_trans)
        L2_dis = torch.norm(warped_src_keypts - tgt_keypts, dim=-1)
        pred_inlier = (L2_dis < inlier_threshold)[0]  # assume bs = 1
        inlier_num = torch.sum(pred_inlier)
        if abs(int(inlier_num - previous_inlier_num)) < 1:
            break
        else:
            previous_inlier_num = inlier_num
        initial_trans = rigid_transform_3d(
            A=src_keypts[:, pred_inlier, :],
            B=tgt_keypts[:, pred_inlier, :],
            ## https://link.springer.com/article/10.1007/s10589-014-9643-2
            # weights=None,
            t = initial_trans[:, :3, 3:4],
            weights=1 / (1 + (L2_dis / inlier_threshold) ** 2)[:, pred_inlier],
            # weights=((1-L2_dis/inlier_threshold)**2)[:, pred_inlier]
        )
    return initial_trans

def test_process(mode, sess, data, cur_global_step, summary_writer, config, xin1, xin2, Rin, tin, fin, is_training,
                 Rhat, that, logits, loss, weights, reg_flag, reg_function):


    import tensorflow as tf

    if mode == 'test':
        print("[{}] {}: Start testing\n".format(config.data_te, time.asctime()))

    if mode == 'valid':
        print("[{}] {}: Start validating\n".format(config.data_te, time.asctime()))

    suffix_tr = config.log_dir
    res_dir_tr = os.path.join(config.res_dir, suffix_tr, "test_result_KITTI")
    if not os.path.exists(res_dir_tr):
        os.makedirs(res_dir_tr)
    # ----------------------------------------
    # Unpack some data for simple coding
    x1 = data['x1']
    x2 = data['x2']
    Rs = data['R']
    ts = data['t']
    fs = data['flag']

    num_samples = len(x1)

    R_hats = []
    t_hats = []
    flags = []
    delta = []
    cls_logs = []
    cls_weights = []
    losses = []
    skews = []
    precisions = []
    accuracies = []

    per = 1
    refine = False

    # for idx_cur in range(100):
    for idx_cur in range(num_samples):

        # Actual construction of the batch
        x1_b = np.array(x1[idx_cur]).reshape((1, 1, -1, 3))
        x1_b = x1_b[:, :, :int(x1_b.shape[2] * per), :]
        x2_b = np.array(x2[idx_cur]).reshape((1, 1, -1, 3))
        x2_b = x2_b[:, :, :int(x2_b.shape[2] * per), :]

        Rs_b = np.array(Rs[idx_cur]).reshape((1, 9))
        ts_b = np.array(ts[idx_cur]).reshape((1, 3))
        fs_b = np.array(fs[idx_cur]).reshape((1,-1))
        fs_b = fs_b[:, :int(fs_b.shape[1] * per)]

        # Feed Dict
        feed_dict = {
            xin1: x1_b,
            xin2: x2_b,
            Rin: Rs_b,
            tin: ts_b,
            fin: fs_b,
            is_training: False,
        }
        fetch = {
            "R_hat": Rhat,
            "t_hat": that,
            "logits": logits,
            "weights": weights,
            "loss": loss,
        }

        time_start = time.time()
        res = sess.run(fetch, feed_dict=feed_dict)
        time_end = time.time() - time_start


        delta.append(time_end)
        if mode == 'test':
            print("[{}] Detection time - {}".format(idx_cur, delta[idx_cur]))

        R_refine = res['R_hat']
        t_refine = res['t_hat']

        R_hats.append(R_refine)
        t_hats.append(t_refine)
        cls_logs.append(res['logits'])
        cls_weights.append(res['weights'])
        losses.append(res['loss'])

        log_pos = np.array(res['logits'] > 0, dtype=np.uint8)
        flags.append(log_pos.astype(bool))

        accuracies.append(metrics.accuracy_score(fs_b[0], log_pos[0]))

    t_errors = []
    r_errors = []
    frob_errors = []
    unused_idx = []

    els = []

    for idx_cur in range(num_samples):
        if reg_flag:

            pts1 = initializePointCoud(x1[idx_cur])
            pts2 = initializePointCoud(x2[idx_cur])

            reg_fun = selectFunction(reg_function)

            if np.asarray(pts1.points).size > 0:

                result, elapsed, pts1_down, pts2_down = globalRegistration(pts1, pts2, reg_fun)
                print('[{}] {} registration took: {}'.format(idx_cur, reg_function, elapsed))
                delta[idx_cur] = elapsed
                e = elapsed

                if reg_function == 'global' and refine:
                    print()
                    transformation, elapsed = refine_registration(pts1_down, pts2_down, result.correspondence_set)
                    print('[{}] {} refinement took: {}'.format(idx_cur, 'Umeyama', elapsed))
                    delta[idx_cur] = delta[idx_cur] + elapsed
                    elapsed = e + elapsed

                    els.append(elapsed)
                    R_hat = transformation[:3,:3]
                    t_hat = transformation[:3, 3]

                else:

                    els.append(elapsed)
                    R_hat = result.transformation[:3, :3]
                    t_hat = result.transformation[:3, 3]


            else:
                print('[{}] IDX registered'.format(idx_cur))
                unused_idx.append(idx_cur)
                R_hat = R_hats[idx_cur].reshape(3, 3)
                t_hat = t_hats[idx_cur]

        else:

            if refine:

                pts1 = initializePointCoud(x1[idx_cur])
                pts2 = initializePointCoud(x2[idx_cur])

                transformation = np.eye(4)
                transformation[:3, :3] = R_hats[idx_cur].reshape(3, 3)
                transformation[:3,  3] = t_hats[idx_cur]

                set = makeMatchesSet(flags[idx_cur])

                transformation, elapsed = refine_registration(pts1, pts2, set)
                print('[{}] {} refinement took: {}'.format(idx_cur, 'Umeyama', elapsed))
                delta[idx_cur] = delta[idx_cur] + elapsed
                els.append(elapsed)

                R_hat = transformation[:3, :3]
                t_hat = transformation[:3, 3]

            else:

                R_hat = R_hats[idx_cur].reshape(3, 3)
                t_hat = t_hats[idx_cur]


                if config.representation == 'linear':
                    u, s, vh = np.linalg.svd(R_hat, compute_uv=True)
                    R_hat = np.matmul(u, vh)
                    R_hat = np.linalg.det(R_hat) * R_hat


        R_gt = Rs[idx_cur].reshape(3,3)
        t_gt = ts[idx_cur]
        fs_cur = fs[idx_cur]
        rot_error, frob_error = RotationError(R_gt, R_hat)
        if np.isnan(rot_error):
            R = np.eye(3)
            rot_error, frob_error = RotationError(R_gt, R)
        if np.isnan(frob_error):
            R = np.eye(3)
            rot_error, frob_error = RotationError(R_gt, R)

        if np.isnan(rot_error):
            print("error")
        else:
            r_errors.append(rot_error)
            frob_errors.append(frob_error)
            t_errors.append(TranslationError(t_hat, t_gt))

    mean_rotation_error = np.mean(np.abs(r_errors))
    mean_translation_error = np.mean(np.abs(t_errors))
    median_rotation_error = np.median(np.abs(r_errors))
    median_translation_error = np.median(np.abs(t_errors))
    mean_frob_error = np.median(np.abs(frob_errors))
    mean_losses = np.mean(np.abs(losses))
    mean_accuracies = np.mean(accuracies)
    mean_deltas = np.mean(delta)

    R_thre = config.R_thre
    t_thre = config.t_thre
    R_gap = R_thre * 2 / 100
    t_gap = t_thre * 2 / 100 / 100 # convert cm to m

    r_error_array = np.array(r_errors)
    r_precision_array = np.zeros((100,), dtype = np.float)
    t_error_array = np.array(t_errors)
    t_precision_array = np.zeros((100,), dtype=np.float)
    rt_precision_array = np.zeros((100,), dtype=np.float)
    for i in range(100):
        r_thre = i * R_gap
        flag = (r_error_array < r_thre).astype(int)
        precision = np.sum(flag) / len(r_errors)
        r_precision_array[i] = precision
        t_thre = i * t_gap
        flag = (t_error_array < t_thre).astype(int)
        precision = np.sum(flag) / len(r_errors)
        t_precision_array[i] = precision
        rt_flag = (r_error_array < r_thre).astype(int) * (t_error_array < t_thre).astype(int)
        precision = np.sum(rt_flag) / len(r_errors)
        rt_precision_array[i] = precision

    with open(os.path.join(res_dir_tr, 'result'), 'a') as f:
        f.write(f'Rotation mean error (deg): {mean_rotation_error:.4f}\n')
        f.write(f'Translation mean error (m): {mean_translation_error:.4f}\n')
        f.write(f'Rotation median error (deg): {median_rotation_error:.4f}\n')
        f.write(f'Translation median error (m): {median_translation_error:.4f}\n')
        f.write(f'Frob norm mean error (deg): {mean_frob_error:.4f}\n')
        f.write(f'Total Loss error: {mean_losses:.4f}\n')
        f.write(f'Accuracies: {mean_accuracies:.4f}\n')
        f.write(f'Mean delta: {mean_deltas:.4f}\n')
        f.write(f'R_recall: {r_precision_array[50]:.4f}\n')
        f.write(f't_recall: {t_precision_array[50]:.4f}\n')
        f.write(f'Rt_Recall: {rt_precision_array[50]:.4f}\n')
        f.write(f'R_mAP: {np.mean(r_precision_array[:50]):.4f}\n')
        f.write(f't_mAP: {np.mean(t_precision_array[:50]):.4f}\n')
        f.write(f'Rt_mAP: {np.mean(rt_precision_array[:50]):.4f}\n')

    summaries = []

    summaries += [
        tf.Summary.Value(tag="ErrorComputation/mean_rotation_error_{}".format(mode),
                         simple_value=mean_rotation_error)]
    summaries += [
        tf.Summary.Value(tag="ErrorComputation/mean_translation_error_{}".format(mode),
                         simple_value=mean_translation_error)]
    summaries += [
        tf.Summary.Value(tag="ErrorComputation/mean_frob_error_{}".format(mode),
                         simple_value=mean_frob_error)]
    summaries += [
        tf.Summary.Value(tag="ErrorComputation/mean_losses_{}".format(mode),
                         simple_value=mean_losses)]
    summaries += [
        tf.Summary.Value(tag="ErrorComputation/mean_accuracies_{}".format(mode),
                         simple_value=mean_accuracies)]

    summary_writer.add_summary(
        tf.Summary(value=summaries), global_step=cur_global_step)

    if mode == 'test':
        # r_errors = np.unwrap(r_errors)
        print('Rotation mean error (deg) - {}'.format(mean_rotation_error))
        print('Translation mean error (m) - {}'.format(mean_translation_error))
        print('Rotation median error (deg) - {}'.format(median_rotation_error))
        print('Translation median error (m) - {}'.format(median_translation_error))
        print('Frob norm mean error (deg) - {}'.format(mean_frob_error))
        print('Total Loss error - {}'.format(mean_losses))
        print('Accuracies - {}'.format(mean_accuracies))
        print('Mean delta - {}'.format(mean_deltas))
        print('Rt_Recall - {}'.format(rt_precision_array[50]))
        print('Rt_mAP - {}'.format(np.mean(rt_precision_array[:50])))

        if els:
            print('Mean elapsed - {}'.format(np.mean(els)))

        # print('Idx min: {} \tRot_min = {}'.format(indx_min, np.abs(r_errors[indx_min])))

        dict = {'RotationError': np.abs(r_errors), 'TranslationError': np.abs(t_errors), 'FrobError': frob_errors,
                'TotalLoss': losses, 'Accuracies': accuracies, 'Delta': delta, 'R_gt': Rs, 't_gt': ts, 'x1': x1,
                'x2': x2, 'flag': flags, 'Rhat': R_hats, 'that': t_hats}

    return mean_frob_error











