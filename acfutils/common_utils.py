from __future__ import division

import math
import os
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torchvision.models.detection.roi_heads import paste_masks_in_image
import matplotlib.cm as cm
import argparse
import sys
import config
from keypoint_3d.kpoint_3d_branch import vote_keypoint_inference, vote_axis_inference, refine_vote_axis
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import cv2

SCORE_THRESH = 0.5
MASK_ALPHA = 0.6


def parse_args(mode):
    if mode != 'train' and mode != 'test':
        print('parse_args argument error')
        exit(0)

    parser = argparse.ArgumentParser(description='')
    if mode == 'train':
        parser.add_argument('--log-freq', default=100, type=int,
                            help='Print log message at this many iterations (default: 100)')
        parser.add_argument('--log-dir', type=str, default='./logs', help='log path')
        parser.add_argument('--epochs', type=int, default=30, metavar='N',
                            help='number of epochs to train (default: 10)')
        parser.add_argument('--resume_epoch', type=int, default=0, metavar='N',
                            help='number of previous epochs to resume on (default: 0)')
        parser.add_argument('--adam', default=True, type=bool, help='use Adam optimizer')
        parser.add_argument('--adam-lr', type=float, default=0.0001, metavar='LR',
                            help='learning rate for Adam (default: 0.01)')
        parser.add_argument('--sgd', default=False, type=bool, help='use SGD optimizer')
        parser.add_argument('--sgd-lr', type=float, default=0.001, metavar='LR',
                            help='learning rate for SGD (default: 0.001)')
        parser.add_argument('--momentum', default=0.9, type=float, help='SGD optimizer momentum')
        parser.add_argument('--weighted-decay', default=1e-4, type=float, help='SGD weighted decay (default: 1e-4)')
        parser.add_argument('--step', default=3, type=int, help='SGD # of epochs per decrease learning rate')

    parser.add_argument('--dataset', type=str, default="unreal_parts")
    parser.add_argument('-d', '--data-dir', default='./data')
    parser.add_argument('--batch-size', type=int, default=2, metavar='N',
                        help='input batch size for training (default: 1)')
    parser.add_argument('--checkpoint-path', default='./weights', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--resume', default= None, type=str,
                        help='resume training from checkpoint-path/model-best.pth')
    parser.add_argument('--input-mode', default='RGB', type=str,
                        help='input mode (RGB or RGBD, default: RGB)')
    parser.add_argument('--acf-head', default='endpoints', type=str,
                        help='the representation of acf')
    parser.add_argument('-j', '--workers', type=int, default=2)
    parser.add_argument('--out-dir', type=str, default='./output', help='output image path')
    args = parser.parse_args()

    print(' '.join(sys.argv))
    print(args)
    return args


def euclidean_distance(c1, c2):
    return np.linalg.norm(c1 - c2, ord=2)

def create_gt_masks(target, img, classes):
    img = img.copy()
    draw = ImageDraw.Draw(img)
    num_classes = len(classes)
    # Colour map for the number of classes.
    colours = cm.get_cmap("jet", num_classes)
    fnt = ImageFont.truetype("FreeSans.ttf", 16)

    num_prop = target['labels'].size(0)

    mask = target['masks'].cpu().numpy()
    boxes = target['boxes'].chunk(num_prop)
    labels = target['labels'].chunk(num_prop)

    for b, l in zip(boxes, labels):
        b, c = (b.squeeze(0).cpu().numpy(), l.squeeze(0).cpu().numpy())
        b = [int(idx) for idx in b]

        # Put the bboxes on the image.
        draw.rectangle(b, outline="#ff0000")
        text_xy = b[:2]
        text_xy[1] = max(text_xy[1] - 18, 0)
        draw.text(text_xy, classes[c], font=fnt, outline="#ff0000")

        # Create a mask with the specified transparency.
        colour_mask = np.uint8(np.where(mask == 255, 0, int(MASK_ALPHA * 255)))
        colour_mask = Image.fromarray(colour_mask).convert("L")

        # Paste the labels onto the image.
        colour_mask_rgb = Image.fromarray(np.uint8(colours(mask.astype(float) / (num_classes - 1)) * 255))
        img.paste(colour_mask_rgb, mask=colour_mask)

    return np.array(img).transpose((2, 0, 1))

def post_process_proposals(proposals, img_depths, img_shape=(960, 1280), K=None, score_thresh=SCORE_THRESH,
                           RGB_only=False, camera_mat = config.Unreal_camera_mat):
    res = []
    for p in proposals:
        scores = p['scores'].cpu()
        masks = p['masks'].cpu()
        boxes = p['boxes'].cpu()
        labels = p['labels'].cpu()
        if not RGB_only:
            keypoints_offset = None
            if 'keypoints_offset' in p.keys():
                keypoints_offset = p['keypoints_offset']
            axis_keypoints_offset = None
            if 'axis_keypoint_offsets' in p.keys():
                axis_keypoints_offset = p['axis_keypoint_offsets']
            axis_offset = None
            if 'axis_offsets' in p.keys():
                axis_offset = p['axis_offsets']

            pafs = None
            if 'pafs' in p.keys():
                pafs = p['pafs'].cpu()

            norm_vectors = None
            if 'norm_vector' in p.keys():
                norm_vectors = p['norm_vector']

        if K is not None:
            scores = scores[:K]
            masks = masks[:K]
            boxes = boxes[:K]
            labels = labels[:K]
            if not RGB_only:
                if keypoints_offset is not None:
                    keypoints_offset = keypoints_offset[:K]
                if axis_keypoints_offset is not None:
                    axis_keypoints_offset = axis_keypoints_offset[:K]
                if pafs is not None:
                    pafs = pafs[:K]

                if axis_offset is not None:
                    axis_offset = axis_offset[:K]
                if norm_vectors is not None:
                    norm_vectors = norm_vectors[:K]

        accepted = torch.nonzero(scores > score_thresh).squeeze()
        if accepted.nelement() == 0:
            res_dict = dict(scores=torch.empty(0),
                            masks=torch.empty(0, *img_shape),
                            boxes=torch.empty(0, 4),
                            labels=torch.empty(0))
            res.append(res_dict)
            continue
        if accepted.dim() == 0:
            accepted = accepted.unsqueeze(0)
        masks = masks[accepted]
        scores = scores[accepted]
        labels = labels[accepted]
        boxes = boxes[accepted]
        if not RGB_only:
            keypoints_3d = None
            center_voters = None
            if keypoints_offset is not None:
                keypoints_offset = keypoints_offset[accepted]
                img_depths = TF.to_tensor(img_depths / 10).type(torch.float32).cuda()
                keypoints_3d, center_voters = vote_keypoint_inference(img_depths, keypoints_offset.cuda(), boxes.cuda(), masks.cuda(), camera_mat)
                keypoints_3d = keypoints_3d.cpu()

            if pafs is not None:
                pafs = pafs[accepted]
                pafs = pafs * masks
                pafs = pafs.sum((2, 3)) / masks.sum((2, 3))
                pafs = F.normalize(pafs, dim=1)

            axis = None
            voters = None
            if axis_keypoints_offset is not None:
                axis_keypoints_offset = axis_keypoints_offset[accepted]
                axis_keypoint1, voters1 = vote_keypoint_inference(img_depths, axis_keypoints_offset[:, 0], boxes.cuda(), masks.cuda(), camera_mat)
                axis_keypoint2, voters2 = vote_keypoint_inference(img_depths, axis_keypoints_offset[:, 1], boxes.cuda(),
                                                        masks.cuda(), camera_mat)
                voters = [voters1, voters2]
                axis_keypoint = torch.stack((axis_keypoint1, axis_keypoint2), dim=1).cpu()
                axis = F.normalize(axis_keypoint[:, 1] - axis_keypoint[:, 0], dim=1)
            elif axis_offset is not None:
                axis_offset = axis_offset[accepted]
                axis, voters, ori_voters = vote_axis_inference(img_depths, axis_offset.cuda(), boxes.cuda(), masks.cuda(), labels)
            elif norm_vectors is not None:
                norm_vectors = norm_vectors[accepted]
                N = masks.shape[0]
                masks_for_vote = F.interpolate(masks, scale_factor=2, mode="bilinear", align_corners=False)
                _, index = torch.sort(masks_for_vote.view(N, -1), dim=-1)
                index = index[:, -1000:]
                estimate_vectors = []
                for i in range(N):
                    norm_v = norm_vectors[i].view(3, -1)[:, index[i]]
                    estimate_vectors.append(norm_v)
                axis = torch.stack(estimate_vectors, dim=0)
                axis = torch.mean(axis, dim=2)
                axis = F.normalize(axis, dim=1)
        masks = paste_masks_in_image(masks, boxes, img_shape, padding=0)

        res.append(dict(scores=scores, masks=masks, boxes=boxes, labels=labels, keypoints_3d=keypoints_3d, axis=axis,
                        axis_voters = voters, center_voters = center_voters, paf_vectors=pafs))
    return res


def vis_images(proposals, targets, image_paths, classes, cam_int_mat, training=True, output_dir=None,
               final_pafs_pair=None, draw_voters=False, save_image_name=None):
    num_classes = len(classes)
    classes_inverse = {classes[key]: key for key in classes}
    # Colour map for the number of classes.
    colours = cm.get_cmap("jet", num_classes)
    fnt = ImageFont.truetype("FreeSans.ttf", 20)

    imgs = []

    for img_index, (p, t, path) in enumerate(zip(proposals, targets, image_paths)):
        # Load the RGB image.
        if type(path) == np.ndarray:
            rgb_img = Image.fromarray(path)
        else:
            rgb_img = Image.open(path).convert("RGB")
        orig = np.array(rgb_img).transpose((2, 0, 1))

        if training:
            gt_mask = create_gt_masks(t, rgb_img, classes)

        num_prop = p['scores'].size(0)

        if num_prop < 1:
            if training:
                imgs.append((orig, gt_mask, orig))
            else:
                imgs.append((orig, 0, orig))
            continue

        # Extract data.
        scores = p['scores'].chunk(num_prop)
        masks = p['masks'].chunk(num_prop)
        boxes = p['boxes'].chunk(num_prop)
        labels = p['labels'].chunk(num_prop)
        keypoints_3d = p['keypoints_3d']
        fx, fy, cx, cy = cam_int_mat[0, 0], cam_int_mat[1, 1], cam_int_mat[0, 2], cam_int_mat[1, 2]

        if keypoints_3d is not None:
            keypoints_x = keypoints_3d[:, 0] / keypoints_3d[:, 2] * fx + cx
            keypoints_y = keypoints_3d[:, 1] / keypoints_3d[:, 2] * fy + cy
            keypoints = torch.stack((keypoints_x, keypoints_y), dim=1).cpu()

            if draw_voters:
                center_voters = p['center_voters']

        axis = p['axis']
        if axis is not None:
            if t is not None:
                target_axis_keypoints = t['axis_keypoints']
                t_keypoints_x = target_axis_keypoints[..., 0] / target_axis_keypoints[..., 2] * fx + cx
                t_keypoints_y = target_axis_keypoints[..., 1] / target_axis_keypoints[..., 2] * fy + cy
                target_axis_kp = torch.stack((t_keypoints_x, t_keypoints_y), dim=2)
            if draw_voters:
                axis_voters = p['axis_voters']

        # for s, m, b, l, p, tp in zip(scores, masks, boxes, labels, poses, target_poses):
        for i in range(len(scores)):
            s, m, b, l = scores[i], masks[i], boxes[i], labels[i]
            s, m, b, l = s.squeeze(0).cpu().numpy(), m.squeeze().cpu().numpy(), b.squeeze(0).cpu().numpy(), \
                         l.cpu().numpy()[0]
            if keypoints is not None:
                k = keypoints[i].squeeze().cpu().numpy()

            # Put the bboxes on the image.
            draw = ImageDraw.Draw(rgb_img)
            draw.rectangle(b, outline="black")
            x1, y1, x2, y2 = b
            text_xy = b[:2]
            text_xy[0] = max((x1 + x2) / 2 - 18, 0)
            if l == 2:
                text_xy[0] = max((x1 + x2) / 2 - 25, 0)
            text_xy[1] = max(y1 - 18, 0)
            draw.text(text_xy, classes_inverse[l], font=fnt, fill='black', outline="black")  # was classes[l] before

            # Create a mask with the specified transparency.
            colour_mask = np.uint8(m * int(MASK_ALPHA * 255))
            colour_mask = Image.fromarray(colour_mask).convert("L")

            # Paste the labels onto the image.
            if l >= 3:
                colour_mask_rgb = Image.fromarray(
                    np.uint8(colours(m.astype(float) * (l + 1) / (num_classes - 1)) * 255))
            else:
                colour_mask_rgb = Image.fromarray(np.uint8(colours(m.astype(float) * l / (num_classes - 1)) * 255))
            rgb_img.paste(colour_mask_rgb, mask=colour_mask)

            rgb_img = cv2.cvtColor(np.asarray(rgb_img), cv2.COLOR_RGB2BGR)
            if keypoints is not None:
                rgb_img = cv2.circle(rgb_img, tuple(k), 4, color=(0, 0, 255), thickness=-1)
                # draw.ellipse([k[0] - 3, k[1] - 3, k[0] + 3, k[1] + 3], fill="red")
                if draw_voters:
                    voter = center_voters[i]
                    keypoints_x = voter[:, 0] / voter[:, 2] * fx + cx
                    keypoints_y = voter[:, 1] / voter[:, 2] * fy + cy
                    r = 1
                    sample = 10
                    for j in range(round(voter.shape[0] / sample)):
                        draw.ellipse(
                            [keypoints_x[sample * j] - r, keypoints_y[sample * j] - r, keypoints_x[sample * j] + r,
                             keypoints_y[sample * j] + r], fill="red")
            if axis is not None:
                length = 10
                one_axis = axis[i].cpu().numpy()
                center_keypoint = keypoints_3d[i].squeeze().cpu().numpy()
                another_keypoint = center_keypoint + length * one_axis
                project_2d_x = another_keypoint[0] / another_keypoint[2] * fx + cx
                project_2d_y = another_keypoint[1] / another_keypoint[2] * fy + cy
                end_point = (int(project_2d_x), int(project_2d_y))
                rgb_img = cv2.arrowedLine(rgb_img, tuple((k).astype("int")), end_point, color=(0, 165, 255), thickness = 3, tipLength=0.2)
                # draw.line([k[0], k[1], project_2d_x, project_2d_y], fill='orange', width=6)
                if draw_voters and axis_voters is not None:
                    if len(axis_voters) == 2:
                        # endpoints voter
                        vote1 = axis_voters[0][i]
                        vote2 = axis_voters[1][i]
                        keypoints_x = vote1[:, 0] / vote1[:, 2] * fx + cx
                        keypoints_y = vote1[:, 1] / vote1[:, 2] * fy + cy
                        keypoints_x2 = vote2[:, 0] / vote2[:, 2] * fx + cx
                        keypoints_y2 = vote2[:, 1] / vote2[:, 2] * fy + cy
                        r = 1
                        sample = 10
                        for j in range(round(vote1.shape[0] / sample)):
                            draw.ellipse(
                                [keypoints_x[sample * j] - r, keypoints_y[sample * j] - r, keypoints_x[sample * j] + r,
                                 keypoints_y[sample * j] + r], fill="orange")
                            draw.ellipse(
                                [keypoints_x2[sample * j] - r, keypoints_y2[sample * j] - r,
                                 keypoints_x2[sample * j] + r,
                                 keypoints_y2[sample * j] + r], fill="pink")
                    else:
                        # scatter voters
                        voter = axis_voters[i]
                        keypoints_x = voter[:, 0] / voter[:, 2] * fx + cx
                        keypoints_y = voter[:, 1] / voter[:, 2] * fy + cy
                        r = 1
                        sample = 1
                        for j in range(round(voter.shape[0] / sample)):
                            if voter[j, 3]:
                                draw.ellipse([keypoints_x[sample * j] - r, keypoints_y[sample * j] - r,
                                              keypoints_x[sample * j] + r, keypoints_y[sample * j] + r], fill="orange")
                            else:
                                draw.ellipse([keypoints_x[sample * j] - r, keypoints_y[sample * j] - r,
                                              keypoints_x[sample * j] + r, keypoints_y[sample * j] + r], fill="pink")
            rgb_img = Image.fromarray(cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB))

        rgb_img = cv2.cvtColor(np.asarray(rgb_img), cv2.COLOR_RGB2BGR)
        if final_pafs_pair is not None:
            for pair in final_pafs_pair[img_index]:
                k1 = keypoints[pair[0]].squeeze().cpu().numpy()
                k2 = keypoints[pair[1]].squeeze().cpu().numpy()
                rgb_img = cv2.line(rgb_img, tuple(k1), tuple(k2), color=(255, 255, 255), thickness=3)

        if axis is not None and t is not None:
            for takp in target_axis_kp:
                takp = takp.cpu().numpy()
                # draw.line([takp[0, 0], takp[0, 1], takp[1, 0], takp[1, 1]], fill='green', width=3)
                start_point = (takp[0, 0], takp[0, 1])
                end_point = (takp[1, 0], takp[1, 1])
                rgb_img = cv2.arrowedLine(rgb_img, start_point, end_point, (0, 255, 0), 3, tipLength=0.2)
            # rgb_img = Image.fromarray(cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB))

        # Repackage data.
        if not training:
            if type(path) is np.ndarray:
                cv2.imwrite(os.path.join(output_dir,save_image_name), rgb_img)
                # rgb_img.save(os.path.join(output_dir,'verp_test.png'))
            else:
                cv2.imwrite(output_dir + path.split('/')[-1].replace('.png', '_{}.png'.format(i)), rgb_img)
                # rgb_img.save(output_dir + path.split('/')[-1].replace('.png', '_{}.png'.format(i)))
                print(output_dir + path.split('/')[-1])
        rgb_img = np.array(rgb_img).transpose((2, 0, 1))
        if training:
            data = (rgb_img, gt_mask, orig)
        else:
            data = (rgb_img, 0, orig)
        imgs.append(data)
    return imgs

def evaluate(proposals, targets, CLASS_LABELS, output_dir):
    pr_stats, keypoint3d_stats = {}, {}  # ADD more if needed
    rot_stats = {}
    for obj in CLASS_LABELS:
        if obj != '__background__':
            pr_stats[obj] = np.zeros((3,))  # TP, FP, FN cases
            keypoint3d_stats[obj] = []
            rot_stats[obj] = []

    print('{} images in total'.format(len(proposals)))
    os.system('mkdir -p ' + output_dir)
    # os.system('mkdir -p ' + output_dir + '/matched_poses_mat/')
    k = 0

    for p, t in zip(proposals, targets):
        labels, keypoints3d, bboxes, axis = p[0], p[1], p[2], p[3]
        target_poses, target_labels= t[0], t[1]
        target_axis_keypoints = t[2]
        # match labels first, within each class, calculate center distance and match pairs, till no gt or estimated pose
        # set remaining gt to fn, estimated to fp; for the matched pairs, calculate ADD, rot, trans errors

        for obj in CLASS_LABELS:
            if obj != '__background__':
                obj_n = CLASS_LABELS[obj]
                target_ind = np.where(target_labels == obj_n)[0]
                estimate_ind = np.where(labels == obj_n)[0]
                while len(target_ind) > 0 and len(estimate_ind) > 0:  # calculate error for a TP
                    estimate_poses_list = keypoints3d[estimate_ind]
                    target_poses_list = target_poses[target_ind]

                    center_dis = np.zeros((len(estimate_poses_list), len(target_poses_list)))
                    for i, p1 in enumerate(estimate_poses_list):
                        for j, p2 in enumerate(target_poses_list):
                            center_dis[i, j] = euclidean_distance(p1, p2[:3])
                    nearest_pair = np.where(center_dis == np.amin(center_dis))
                    matched_estimate_ind, matched_target_ind = estimate_ind[nearest_pair[0][0]], target_ind[nearest_pair[1][0]]
                    estimate_pose = keypoints3d[matched_estimate_ind]
                    target_pose = target_poses[matched_target_ind]
                    keypoint3d_stats[obj].append(euclidean_distance(estimate_pose, target_pose[:3]))

                    pr_stats[obj][0] += 1
                    estimate_axis = axis[matched_estimate_ind]
                    target_akp = target_axis_keypoints[matched_target_ind]
                    target_axis = (target_akp[1]-target_akp[0])/np.linalg.norm(target_akp[1]-target_akp[0])
                    v = 180.0 / math.pi * math.acos(np.clip(np.dot(estimate_axis, target_axis), -1 + 1e-9, 1 - 1e-9))
                    rot_stats[obj].append(v)

                    # remove both from list, update index
                    labels = np.delete(labels, matched_estimate_ind)
                    target_poses = np.delete(target_poses, matched_target_ind, 0)
                    target_labels = np.delete(target_labels, matched_target_ind)
                    bboxes = np.delete(bboxes, matched_estimate_ind, 0)
                    keypoints3d = np.delete(keypoints3d, matched_estimate_ind, 0)
                    axis = np.delete(axis, matched_estimate_ind, 0)
                    target_axis_keypoints = np.delete(target_axis_keypoints, matched_target_ind, 0)

                    target_ind = np.where(target_labels == obj_n)[0]
                    estimate_ind = np.where(labels == obj_n)[0]


                if len(target_ind) > 0:  # add to stats FN
                    pr_stats[obj][2] += len(target_ind)
                if len(estimate_ind) > 0:  # add to stats FP
                    pr_stats[obj][1] += len(estimate_ind)
        print(k)
        k += 1

    with open(output_dir + 'stats.txt', 'w') as f:
        f.write("obj, #TP, #FP, #FN\n")
        for obj in pr_stats.keys():
            f.write("%s  %d  %d  %d\n" % (obj, pr_stats[obj][0], pr_stats[obj][1], pr_stats[obj][2]))

    return {'translation': keypoint3d_stats, 'Nstats': pr_stats, 'rotation': rot_stats}


def plot_auc(results_stats, output_dir):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    import matplotlib.pyplot as plt

    object_color = {'body': 'red',
                    'head': 'green',
                    'handle': 'blue',
                    'stir': 'black'}
    translation_thres = [2, 5]
    rotation_thres = [5, 10, 15]

    with open(output_dir + 'auc_parts.txt', 'w') as f:
        for t_thres in translation_thres:
            for r_thres in rotation_thres:
                total_auc = 0
                f.write("translation %s cm, rotation %s degree : \n" % (t_thres, r_thres))
                for obj_index, obj in enumerate(results_stats['Nstats'].keys()):
                    TP = results_stats['Nstats'][obj][0]
                    a = np.array(results_stats['rotation'][obj]) < r_thres
                    b = np.array(results_stats['translation'][obj]) < t_thres
                    auc = np.sum(a*b)/TP
                    total_auc += auc
                    f.write("%s : %.3f \n"%(obj, auc))
                f.write("mean : %.3f \n"%(total_auc/4))
    r_toal_values, r_base = 0, 0
    t_toal_values, t_base = 0, 0
    for obj_index, obj in enumerate(results_stats['Nstats'].keys()):
        TP, FP, FN = results_stats['Nstats'][obj][0], results_stats['Nstats'][obj][1], results_stats['Nstats'][obj][2]

        plt.figure(0)
        values, r_base = np.histogram(np.array(results_stats['rotation'][obj]), range=(0, 180), bins=90)
        cumulative = np.cumsum(np.insert(values, 0, 0)) / TP
        r_toal_values += cumulative
        # plot the cumulative function
        plt.plot(r_base, cumulative, c=object_color[obj], label=obj)
        plt.legend()
        plt.grid(True)
        plt.xlim([0, 180])
        plt.ylim([0, 1])
        plt.title('Rotation Accuracy Curve')
        plt.xlabel('Degrees')
        plt.ylabel('Accuracy')
        plt.savefig(output_dir + '/rotation_auc.png')

        plt.figure(1)
        values, t_base = np.histogram(np.array(results_stats['translation'][obj]), range=(0, 10), bins=20)
        cumulative = np.cumsum(np.insert(values, 0, 0)) / TP
        t_toal_values += cumulative
        # plot the cumulative function
        plt.plot(t_base, cumulative, c=object_color[obj], label=obj)
        plt.legend()
        plt.grid(True)
        plt.xlim([0, 10])
        plt.ylim([0, 1])
        plt.title('Translation Accuracy Curve')
        plt.xlabel('Centimeter')
        plt.ylabel('Accuracy')
        plt.savefig(output_dir + '/translation_auc.png')

    plt.figure(0)
    plt.plot(r_base, r_toal_values/4, c='orange', label='mean')
    plt.legend()
    plt.grid(True)
    plt.xlim([0, 180])
    plt.ylim([0, 1])
    plt.title('Rotation Accuracy Curve')
    plt.xlabel('Degrees')
    plt.ylabel('Accuracy')
    plt.savefig(output_dir + '/rotation_auc.png')

    plt.figure(1)
    plt.plot(t_base, t_toal_values/4, c='orange', label='mean')
    plt.legend()
    plt.grid(True)
    plt.xlim([0, 10])
    plt.ylim([0, 1])
    plt.title('Translation Accuracy Curve')
    plt.xlabel('Centimeter')
    plt.ylabel('Accuracy')
    plt.savefig(output_dir + '/translation_auc.png')

    import pickle
    f = open(output_dir +'testdata.pickle', 'wb')
    pickle.dump(results_stats, f)
    f.close()

def pafprocess(proposals, camera_m):
    fx, fy, cx, cy = camera_m[0, 0], camera_m[1, 1], camera_m[0, 2], camera_m[1, 2]
    paf_pairs = [[1, 2, 0], [3, 4, 2], [4, 3, 2]] # mug:0, bottle:1, scoop:2
    pair_threshold = 0.5
    distance_threshold = 20
    final_pairs = []
    for p in proposals:
        keypoints_3d = p['keypoints_3d']
        keypoints_x = keypoints_3d[:, 0] / keypoints_3d[:, 2] * fx + cx
        keypoints_y = keypoints_3d[:, 1] / keypoints_3d[:, 2] * fy + cy
        keypoints = torch.stack((keypoints_x, keypoints_y), dim=1).cpu().numpy()
        keypoints_3d = keypoints_3d.cpu().numpy()
        pafs = p['paf_vectors'].numpy()
        label = p['labels'].numpy()
        candidate_pair = []
        for pair in paf_pairs:
            label_a, label_b = pair[0], pair[1]
            obj_name = pair[2]
            index_a = np.where(label == label_a)[0]
            index_b = np.where(label == label_b)[0]
            while index_a.size > 0 and index_b.size > 0:
                paf_errors = np.zeros((index_a.size, index_b.size))
                distances = np.zeros((index_a.size, index_b.size))
                for i in range(index_a.size):
                    for j in range(index_b.size):
                        keypoint_vector = keypoints[index_b[j]] - keypoints[index_a[i]]
                        keypoint_vector = keypoint_vector/np.linalg.norm(keypoint_vector)
                        distance = np.linalg.norm(keypoints_3d[index_b[j]] - keypoints_3d[index_a[i]])
                        paf_vector = pafs[index_a[i]]
                        paf_errors[i, j] = np.dot(keypoint_vector, paf_vector)
                        distances[i, j] = distance
                totoal_error = paf_errors-0.01*distances
                max_value = np.max(totoal_error)
                nearest_pair = np.where(totoal_error == max_value)
                corresponding_distance = distances[nearest_pair]
                score = paf_errors[nearest_pair]
                if corresponding_distance<=distance_threshold:
                    matched_ind_a, matched_ind_b = index_a[nearest_pair[0][0]], index_b[nearest_pair[1][0]]
                    candidate_pair.append([matched_ind_a, matched_ind_b, score[0], obj_name])

                    index_a = np.delete(index_a, nearest_pair[0][0])
                    index_b = np.delete(index_b, nearest_pair[1][0])
                else:
                    index_b = np.delete(index_b, nearest_pair[1][0])

        final_pair = []
        candidate_pair.sort(key=lambda x: x[2], reverse = True)
        for i in range(len(candidate_pair)):
            candidate = candidate_pair[i]
            assigned = False
            if len(final_pair)>0:
                for j in range(len(final_pair)):
                    conn = final_pair[j]
                    if (candidate[0] in conn[:2]) or (candidate[1] in conn[:2]):
                        assigned = True
                        break
            if assigned:
                continue
            final_pair.append(candidate)
        final_pairs.append(final_pair)
    return final_pairs