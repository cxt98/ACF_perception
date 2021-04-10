import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.ops import boxes as box_ops
from torchvision.models.detection.faster_rcnn import TwoMLPHead, FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNHeads, MaskRCNNPredictor
from torchvision.models.detection.roi_heads import fastrcnn_loss, maskrcnn_inference, project_masks_on_boxes
from torchvision.models.detection import _utils as det_utils
from torchvision.ops import MultiScaleRoIAlign
from torchvision.ops import roi_align
import numpy as np

from keypoint_3d.kpoint_3d_branch import ContextBlock, Vote_Kpoints_head, Vote_Kpoints_Predictor
import config


class RoIHeadsExtend(nn.Module):
    def __init__(self, out_channels, num_classes, input_mode, acf_head,
                 fg_iou_thresh=0.5, bg_iou_thresh=0.5, batch_size_per_image=512,
                 positive_fraction=0.25, bbox_reg_weights=None,
                 box_score_thresh=0.05, box_nms_thresh=0.5, box_detections_per_img=100):
        super(RoIHeadsExtend, self).__init__()

        self.in_channels = out_channels
        self.input_mode = input_mode
        self.score_thresh = box_score_thresh
        self.nms_thresh = box_nms_thresh
        self.detections_per_img = box_detections_per_img
        self.fg_iou_thresh = fg_iou_thresh
        self.bg_iou_thresh = bg_iou_thresh
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction
        self.num_classes = num_classes

        # Detection
        self.box_similarity = box_ops.box_iou
        # assign ground-truth boxes for each proposal
        self.proposal_matcher = det_utils.Matcher(
            fg_iou_thresh,
            bg_iou_thresh,
            allow_low_quality_matches=False)

        self.fg_bg_sampler = det_utils.BalancedPositiveNegativeSampler(
            batch_size_per_image,
            positive_fraction)

        if bbox_reg_weights is None:
            bbox_reg_weights = (10., 10., 5., 5.)
        self.box_coder = det_utils.BoxCoder(bbox_reg_weights)

        self.box_roi_pool = MultiScaleRoIAlign(
            featmap_names=[0, 1, 2, 3],
            output_size=7,
            sampling_ratio=2)

        representation_size = 1024
        resolution = self.box_roi_pool.output_size[0]
        self.box_head = TwoMLPHead(
            out_channels * resolution ** 2,
            representation_size)

        self.box_predictor = FastRCNNPredictor(
            representation_size,
            num_classes)

        # Segmentation
        self.shared_roi_pool = MultiScaleRoIAlign(
            featmap_names=[0, 1, 2, 3],
            output_size=14,
            sampling_ratio=2)
        resolution = self.shared_roi_pool.output_size[0]

        mask_layers = (256, 256, 256, 256, 256, 256, 256, 256)
        mask_dilation = 1
        self.mask_head = MaskRCNNHeads(out_channels, mask_layers, mask_dilation)

        mask_predictor_in_channels = 256  # == mask_layers[-1]
        mask_dim_reduced = 256
        self.mask_predictor = MaskRCNNPredictor(mask_predictor_in_channels, mask_dim_reduced, num_classes)

        self.with_paf_branch = True
        if self.with_paf_branch:
            self.paf_head = MaskRCNNHeads(out_channels, mask_layers, mask_dilation)
            self.paf_predictor = MaskRCNNPredictor(mask_predictor_in_channels, mask_dim_reduced, 2*(num_classes-1))

        if self.input_mode == config.INPUT_RGBD:
            self.attention_block = ContextBlock(256, 2)
            self.global_feature_dim = 256
            self.with_3d_keypoints = True
            self.with_axis_keypoints = False
            self.regress_axis = False
            self.estimate_norm_vector = False
            if acf_head == 'endpoints':
                self.with_axis_keypoints = True
            elif acf_head == 'scatters':
                self.regress_axis = True
            elif acf_head == 'norm_vector':
                self.estimate_norm_vector = True
            else:
                print("Don't assign a vaild acf head")
                exit()
            keypoint_layers = (256,)*4
            self.keypoint_dim_reduced = keypoint_layers[-1]
            if self.with_3d_keypoints:
                self.vote_keypoint_head = Vote_Kpoints_head(self.global_feature_dim, keypoint_layers, "conv2d")
                self.vote_keypoint_predictor = Vote_Kpoints_Predictor(self.keypoint_dim_reduced, 3*(num_classes-1))
            if self.with_axis_keypoints:
                self.orientation_keypoint_head = Vote_Kpoints_head(self.global_feature_dim, keypoint_layers, "conv2d")

                self.orientation_keypoint_predictor = Vote_Kpoints_Predictor(self.keypoint_dim_reduced, 6*(num_classes-1))

            if self.regress_axis:
                self.axis_head = Vote_Kpoints_head(self.global_feature_dim, keypoint_layers, "conv2d")
                self.axis_predictor = Vote_Kpoints_Predictor(self.keypoint_dim_reduced, 4 * (num_classes - 1))

            if self.estimate_norm_vector:
                self.norm_vector_head = Vote_Kpoints_head(self.global_feature_dim, keypoint_layers, "conv2d")
                self.norm_vector_predictor = Vote_Kpoints_Predictor(self.keypoint_dim_reduced, 3 * (num_classes - 1))

    # below functions copied from torchvision/models/detection/roi_heads.py
    @property
    def has_mask(self):
        if self.shared_roi_pool is None:
            return False
        if self.mask_head is None:
            return False
        if self.mask_predictor is None:
            return False
        return True

    def assign_targets_to_proposals(self, proposals, gt_boxes, gt_labels):
        matched_idxs = []
        labels = []
        for proposals_in_image, gt_boxes_in_image, gt_labels_in_image in zip(proposals, gt_boxes, gt_labels):
            match_quality_matrix = self.box_similarity(gt_boxes_in_image, proposals_in_image)
            matched_idxs_in_image = self.proposal_matcher(match_quality_matrix)

            clamped_matched_idxs_in_image = matched_idxs_in_image.clamp(min=0)

            labels_in_image = gt_labels_in_image[clamped_matched_idxs_in_image]
            labels_in_image = labels_in_image.to(dtype=torch.int64)

            # Label background (below the low threshold)
            bg_inds = matched_idxs_in_image == self.proposal_matcher.BELOW_LOW_THRESHOLD
            labels_in_image[bg_inds] = 0

            # Label ignore proposals (between low and high thresholds)
            ignore_inds = matched_idxs_in_image == self.proposal_matcher.BETWEEN_THRESHOLDS
            labels_in_image[ignore_inds] = -1  # -1 is ignored by sampler

            matched_idxs.append(clamped_matched_idxs_in_image)
            labels.append(labels_in_image)
        return matched_idxs, labels

    def subsample(self, labels):
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        sampled_inds = []
        for img_idx, (pos_inds_img, neg_inds_img) in enumerate(
                zip(sampled_pos_inds, sampled_neg_inds)
        ):
            img_sampled_inds = torch.nonzero(pos_inds_img | neg_inds_img).squeeze(1)
            sampled_inds.append(img_sampled_inds)
        return sampled_inds

    def add_gt_proposals(self, proposals, gt_boxes):
        proposals = [
            torch.cat((proposal, gt_box))
            for proposal, gt_box in zip(proposals, gt_boxes)
        ]

        return proposals

    def check_targets(self, targets):
        assert targets is not None
        assert all("boxes" in t for t in targets)
        assert all("labels" in t for t in targets)
        if self.has_mask:
            assert all("masks" in t for t in targets)

    def select_training_samples(self, proposals, targets):
        self.check_targets(targets)
        gt_boxes = [t["boxes"] for t in targets]
        gt_labels = [t["labels"] for t in targets]

        # append ground-truth bboxes to propos
        proposals = self.add_gt_proposals(proposals, gt_boxes)

        # get matching gt indices for each proposal
        matched_idxs, labels = self.assign_targets_to_proposals(proposals, gt_boxes, gt_labels)
        # sample a fixed proportion of positive-negative proposals
        sampled_inds = self.subsample(labels)
        matched_gt_boxes = []
        num_images = len(proposals)
        for img_id in range(num_images):
            img_sampled_inds = sampled_inds[img_id]
            proposals[img_id] = proposals[img_id][img_sampled_inds]
            labels[img_id] = labels[img_id][img_sampled_inds]
            matched_idxs[img_id] = matched_idxs[img_id][img_sampled_inds]
            matched_gt_boxes.append(gt_boxes[img_id][matched_idxs[img_id]])

        regression_targets = self.box_coder.encode(matched_gt_boxes, proposals)
        return proposals, matched_idxs, labels, regression_targets

    def postprocess_detections(self, class_logits, box_regression, proposals, image_shapes):
        device = class_logits.device
        num_classes = class_logits.shape[-1]

        boxes_per_image = [len(boxes_in_image) for boxes_in_image in proposals]
        pred_boxes = self.box_coder.decode(box_regression, proposals)

        pred_scores = F.softmax(class_logits, -1)

        # split boxes and scores per image
        pred_boxes = pred_boxes.split(boxes_per_image, 0)
        pred_scores = pred_scores.split(boxes_per_image, 0)

        all_boxes = []
        all_scores = []
        all_labels = []
        for boxes, scores, image_shape in zip(pred_boxes, pred_scores, image_shapes):
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            # create labels for each prediction
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, -1).expand_as(scores)

            # remove predictions with the background label
            boxes = boxes[:, 1:]
            scores = scores[:, 1:]
            labels = labels[:, 1:]

            # batch everything, by making every class prediction be a separate instance
            boxes = boxes.reshape(-1, 4)
            scores = scores.flatten()
            labels = labels.flatten()

            # remove low scoring boxes
            inds = torch.nonzero(scores > self.score_thresh).squeeze(1)
            boxes, scores, labels = boxes[inds], scores[inds], labels[inds]

            # non-maximum suppression, independently done per class
            keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)
            # keep only topk scoring predictions
            keep = keep[:self.detections_per_img]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)

        return all_boxes, all_scores, all_labels

    def forward(self, features, proposals, image_shapes, targets=None):
        """
        Arguments:
            features (List[Tensor])
            proposals (List[Tensor[N, 4]])
            image_shapes (List[Tuple[H, W]])
            targets (List[Dict])
        """
        feature_dims = np.array([features[layer].shape[1] for layer in features])
        if np.all(feature_dims == self.in_channels):  # RGB only
            features_rgb = features
        elif np.all(feature_dims == 2 * self.in_channels):  # RGB-depth 6 channel, two backbones
            from collections import OrderedDict
            features_rgb = OrderedDict()
            for key in features.keys():
                features_rgb[key] = features[key][:, :self.in_channels]
        else: # RGB-D 4 channel
            features_rgb = features

        # Detection
        if self.training:
            proposals, matched_idxs, labels, regression_targets = self.select_training_samples(proposals, targets)

        box_features = self.box_roi_pool(features_rgb, proposals, image_shapes)
        box_features = self.box_head(box_features)
        class_logits, box_regression = self.box_predictor(box_features)

        result, losses = [], {}
        if self.training:
            loss_classifier, loss_box_reg = fastrcnn_loss(
                class_logits, box_regression, labels, regression_targets)
            losses = dict(loss_classifier=loss_classifier, loss_box_reg=loss_box_reg)
        else:
            boxes, scores, labels = self.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
            num_images = len(boxes)
            for i in range(num_images):
                if boxes[i].shape[0] == 0:
                    return result, losses
                result.append(
                    dict(
                        boxes=boxes[i],
                        labels=labels[i],
                        scores=scores[i],
                    )
                )

        # Proposals selected by detection stage is shared by all other branches
        box_proposals = [p["boxes"] for p in result]
        if self.training:
            # during training, only focus on positive boxes
            num_images = len(proposals)
            box_proposals = []
            pos_matched_idxs = []
            for img_id in range(num_images):
                pos = torch.nonzero(labels[img_id] > 0).squeeze(1)
                box_proposals.append(proposals[img_id][pos])
                pos_matched_idxs.append(matched_idxs[img_id][pos])

        # MultiStage RoIAlign is shared by all other branches
        shared_features_rgb = self.shared_roi_pool(features_rgb, box_proposals, image_shapes)

        # Segmentation
        mask_features = self.mask_head(shared_features_rgb)
        mask_logits = self.mask_predictor(mask_features)
        loss_mask = {}
        masks_on_features = None
        if self.training:
            gt_masks = [t["masks"] for t in targets]
            gt_labels = [t["labels"] for t in targets]
            gt_is = [t["instance_ids"] for t in targets]
            loss_mask, masks_for_paf, masks_for_vote = maskrcnn_loss_updated(
                mask_logits, box_proposals,
                gt_masks, gt_labels, gt_is, pos_matched_idxs)
            loss_mask = dict(loss_mask=loss_mask)
        else:
            ref_labels = [r["labels"] for r in result]
            masks_probs = maskrcnn_inference(mask_logits, ref_labels)
            for mask_prob, r in zip(masks_probs, result):
                r["masks"] = mask_prob

        losses.update(loss_mask)

        if self.with_paf_branch:
            paf_features = self.paf_head(shared_features_rgb)
            paf_logits = self.paf_predictor(paf_features)
            loss_paf = {}
            if self.training:
                gt_pafs = [t["target_pafs"] for t in targets]
                loss_paf = paf_loss_updated(paf_logits, masks_for_paf, pos_matched_idxs, gt_pafs, gt_labels)
                if torch.isnan(loss_paf):
                    print('error')
                loss_paf = dict(loss_paf=loss_paf)
            else:
                paf_ref_labels = torch.cat(ref_labels) - 1
                N, _, H, W = paf_logits.shape
                paf_logits = paf_logits.view(N, -1, 2, H, W)[torch.arange(N), paf_ref_labels]
                paf_probs = [paf_logits]
                for paf_prob, r in zip(paf_probs, result):
                    r["pafs"] = F.normalize(paf_prob, dim=1)

            losses.update(loss_paf)

        if self.input_mode == config.INPUT_RGBD:

            shared_features = self.attention_block(shared_features_rgb)  # shared_features_rgb actually has 4-channel RGBD input
            bs, c, _, _ = shared_features.shape
            # shared_features = shared_features.view(bs, c, -1) # for conv1d
            if self.with_3d_keypoints:

                keypoint_features = self.vote_keypoint_head(shared_features)
                # keypoint_features = keypoint_features.view(bs, self.keypoint_dim_reduced, 14, 14)
                keypoint_offsets = self.vote_keypoint_predictor(keypoint_features)

                loss_keypoint = {}
                if self.training:
                    gt_3d_keypoints = [t["frame"][:, :3] for t in targets]
                    ori_depth = [t["ori_image_depth"] for t in targets]
                    gt_labels = [t["labels"] for t in targets]
                    loss_keypoint = vote_keypoint_loss(keypoint_offsets, box_proposals, ori_depth, gt_3d_keypoints,
                                                       pos_matched_idxs, masks_for_vote, gt_labels)
                    loss_keypoint = dict(loss_keypoint=loss_keypoint)
                else:
                    ref_labels = torch.cat(ref_labels) - 1
                    N, _, H, W = keypoint_offsets.shape
                    keypoint_offsets = keypoint_offsets.view(N, -1, 3, H, W)[torch.arange(N), ref_labels]
                    keypoints = [keypoint_offsets]
                    for kps, r in zip(keypoints, result):
                        r["keypoints_offset"] = kps
                losses.update(loss_keypoint)

            if self.with_axis_keypoints:

                keypoint_features = self.orientation_keypoint_head(shared_features)
                # keypoint_features = keypoint_features.view(bs, self.keypoint_dim_reduced, 14, 14)
                axis_keypoint_offsets = self.orientation_keypoint_predictor(keypoint_features)
                N, _, H, W = axis_keypoint_offsets.shape
                axis_keypoint_offsets = axis_keypoint_offsets.view(N, -1, 2, 3, H, W)

                loss_orientation = {}
                if self.training:
                    gt_3d_keypoints = [t["axis_keypoints"] for t in targets]
                    ori_depth = [t["ori_image_depth"] for t in targets]
                    loss_orientation = vote_orientation_loss(axis_keypoint_offsets, box_proposals, ori_depth, gt_3d_keypoints,
                                                             pos_matched_idxs, masks_for_vote, gt_labels)
                    loss_orientation = dict(loss_orientation=loss_orientation)
                else:
                    axis_keypoint_offsets = axis_keypoint_offsets[torch.arange(N), ref_labels]
                    axis_keypoints = [axis_keypoint_offsets]
                    for kps, r in zip(axis_keypoints, result):
                        r["axis_keypoint_offsets"] = kps

                losses.update(loss_orientation)

            if self.regress_axis:
                keypoint_features = self.axis_head(shared_features)
                keypoint_features = keypoint_features.view(bs, self.keypoint_dim_reduced, 14, 14)
                axis_keypoint_offsets = self.axis_predictor(keypoint_features)
                N, _, H, W = axis_keypoint_offsets.shape
                axis_keypoint_offsets = axis_keypoint_offsets.view(N, -1, 4, H, W)

                loss_axis = {}
                if self.training:
                    gt_3d_keypoints = [t["axis_keypoints"] for t in targets]
                    ori_depth = [t["ori_image_depth"] for t in targets]
                    loss_axis = vote_axis_loss(axis_keypoint_offsets, box_proposals, ori_depth,
                                                             gt_3d_keypoints,
                                                             pos_matched_idxs, masks_for_vote, gt_labels)
                    loss_axis = dict(loss_axis=loss_axis)
                else:
                    axis_keypoint_offsets = axis_keypoint_offsets[torch.arange(N), ref_labels]
                    axis_keypoints = [axis_keypoint_offsets]
                    for kps, r in zip(axis_keypoints, result):
                        r["axis_offsets"] = kps

                losses.update(loss_axis)

            if self.estimate_norm_vector:
                keypoint_features = self.norm_vector_head(shared_features)
                keypoint_features = keypoint_features.view(bs, self.keypoint_dim_reduced, 14, 14)
                norm_vectors = self.norm_vector_predictor(keypoint_features)
                N, _, H, W = norm_vectors.shape
                norm_vectors = norm_vectors.view(N, -1, 3, H, W)

                loss_norm_vector = {}
                if self.training:
                    gt_3d_keypoints = [t["axis_keypoints"] for t in targets]
                    loss_norm_vector = calculate_norm_vectors(norm_vectors, gt_3d_keypoints, pos_matched_idxs, masks_for_vote, gt_labels)
                    loss_norm_vector = dict(loss_norm_vector = loss_norm_vector)
                else:
                    norm_vectors = F.normalize(norm_vectors, dim=2)
                    norm_vectors = norm_vectors[torch.arange(N), ref_labels]
                    estimate_norm_vectors = [norm_vectors]
                    for norm_v, r in zip(estimate_norm_vectors, result):
                        r["norm_vector"] = norm_v

                losses.update(loss_norm_vector)
        return result, losses


# below 2 functions are slighted different from Pytorch library: masks not given as binary, they are selected based on
# cropping semantic segmentation labels by bounding boxes
def project_masks_on_boxes(gt_masks, boxes, matched_idxs, gt_labels, M):
    """
    Given segmentation masks and the bounding boxes corresponding
    to the location of the masks in the image, this function
    crops and resizes the masks in the position defined by the
    boxes. This prepares the masks for them to be fed to the
    loss computation as the targets.
    """
    bin_masks = gt_masks.unsqueeze(-1) == gt_labels
    bin_masks = bin_masks.permute(2, 0, 1)
    matched_idxs = matched_idxs.to(boxes)
    rois = torch.cat([matched_idxs[:, None], boxes], dim=1)
    bin_masks = bin_masks[:, None].to(rois)
    return roi_align(bin_masks, rois, (M, M), 1)[:, 0]


def maskrcnn_loss_updated(mask_logits, proposals, gt_masks, gt_labels, gt_is, mask_matched_idxs):
    """
    Arguments:
        proposals (list[BoxList])
        mask_logits (Tensor)
        targets (list[BoxList])
    Return:
        mask_loss (Tensor): scalar tensor containing the loss
    """
    K, C, H, W = mask_logits.shape

    discretization_size = mask_logits.shape[-1]
    labels = [l[idxs] for l, idxs in zip(gt_labels, mask_matched_idxs)]
    mask_targets = [
        project_masks_on_boxes(m, p, i, l, discretization_size)
        for m, p, i, l in zip(gt_masks, proposals, mask_matched_idxs, gt_is)
    ]

    labels = torch.cat(labels, dim=0)
    mask_targets = torch.cat(mask_targets, dim=0)

    mask_targets_for_vote = [
        project_masks_on_boxes(m, p, i, l, 56)
        for m, p, i, l in zip(gt_masks, proposals, mask_matched_idxs, gt_is)
    ]
    mask_targets_for_vote = torch.cat(mask_targets_for_vote, dim=0)

    # torch.mean (in binary_cross_entropy_with_logits) doesn't
    # accept empty tensors, so handle it separately
    if mask_targets.numel() == 0:
        return mask_logits.sum() * 0

    mask_loss = F.binary_cross_entropy_with_logits(
        mask_logits[torch.arange(labels.shape[0], device=labels.device), labels], mask_targets
    )
    return mask_loss, mask_targets, mask_targets_for_vote

def paf_loss_updated(paf_logits, gt_masks, mask_matched_idxs, gt_pafs, gt_labels):
    N, _, H, W = paf_logits.shape
    target_pafs = []
    target_labels = []
    for paf, index, label in zip(gt_pafs, mask_matched_idxs, gt_labels):
        target_paf = paf[index]
        target_label = label[index] - 1
        target_pafs.append(target_paf)
        target_labels.append(target_label)

    target_labels = torch.cat(target_labels, dim=0)
    paf_logits = paf_logits.view(N, -1, 2, H, W)[torch.arange(N), target_labels]
    criterion = nn.MSELoss(reduction='mean').cuda()
    target_pafs = torch.cat(target_pafs, dim=0)
    vaild_index = (target_pafs[:, 0]**2+target_pafs[:, 1]**2) > 0.5
    target = target_pafs.view(N, 2, 1, 1).repeat(1, 1, H, W) * gt_masks.view(N, 1, H, W).repeat(1, 2, 1, 1)
    paf_logits = F.normalize(paf_logits, dim=1) * gt_masks.view(N, 1, H, W).repeat(1, 2, 1, 1)
    loss = criterion(paf_logits[vaild_index], target[vaild_index])

    return loss

def gt_voters(proposal_height, depth_image, bbox, camera_matrix):
    device = bbox.device

    bs, _ = bbox.shape
    depth_image = depth_image.squeeze()
    fx, cx, fy, cy = camera_matrix[0, 0], camera_matrix[0, 2], camera_matrix[1, 1], camera_matrix[1, 2]
    offset_x = bbox[:, 0]
    offset_y = bbox[:, 1]

    scale_x = (bbox[:, 2] - bbox[:, 0]) / proposal_height
    scale_y = (bbox[:, 3] - bbox[:, 1]) / proposal_height
    max_x_index = torch.tensor(depth_image.shape[1]-1).view(1, 1, 1).long().to(device)
    max_y_index = torch.tensor(depth_image.shape[0]-1).view(1, 1, 1).long().to(device)
    max_x_index = max_x_index.repeat(bs, proposal_height, proposal_height)
    max_y_index = max_y_index.repeat(bs, proposal_height, proposal_height)

    dst_y, dst_x = torch.meshgrid(torch.arange(proposal_height), torch.arange(proposal_height))
    dst_x = dst_x.view(1, proposal_height, proposal_height).repeat(bs, 1, 1).to(device)
    dst_y = dst_y.view(1, proposal_height, proposal_height).repeat(bs, 1, 1).to(device)
    src_x = dst_x.float() * scale_x.view(bs, 1, 1).repeat(1, proposal_height, proposal_height)+offset_x.view(
        bs, 1, 1).repeat(1, proposal_height,proposal_height)
    src_y = dst_y.float() * scale_y.view(bs, 1, 1).repeat(1, proposal_height, proposal_height) + offset_y.view(
        bs, 1, 1).repeat(1, proposal_height, proposal_height)
    src_x_0 = torch.floor(src_x).long()
    src_y_0 = torch.floor(src_y).long()
    src_x_1 = torch.min(src_x_0 + 1, max_x_index)
    src_y_1 = torch.min(src_y_0 + 1, max_y_index)

    value0 = (src_x_1.float() - src_x) * depth_image[src_y_0, src_x_0] + (src_x - src_x_0.float()) * depth_image[src_y_0, src_x_1]
    value1 = (src_x_1.float() - src_x) * depth_image[src_y_1, src_x_0] + (src_x - src_x_0.float()) * depth_image[src_y_1, src_x_1]
    depth_value = (src_y_1.float() - src_y) * value0 + (src_y - src_y_0.float()) * value1

    x = (src_x - cx) * depth_value / fx
    y = (src_y - cy) * depth_value / fy

    voter = torch.stack((x, y, depth_value), dim=1)
    return voter


def vote_keypoint_loss(keypoint_offsets, proposals, ori_depths, gt_keypoints, keypoint_matched_idxs, masks_for_vote, gt_labels):
    N, K, H, W = keypoint_offsets.shape
    assert H == W
    vote_offset_targets = []
    voter_labels = []
    for proposals_per_image, gt_kp_in_image, midx, ori_depth, label in zip(proposals, gt_keypoints, keypoint_matched_idxs,
                                                                    ori_depths, gt_labels):
        kp = gt_kp_in_image[midx]
        voter = gt_voters(H, ori_depth, proposals_per_image, config.Unreal_camera_mat)
        gt_offset = kp.view(-1, 3, 1, 1).repeat(1, 1, H, W) - voter
        vote_offset_targets.append(gt_offset)
        voter_labels.append(label[midx]-1)

    voter_labels = torch.cat(voter_labels, dim= 0)
    keypoint_offsets = keypoint_offsets.view(N, -1, 3, H, W)[torch.arange(N), voter_labels]
    votes_targets = torch.cat(vote_offset_targets, dim=0)
    keypoint_loss = torch.abs(keypoint_offsets - votes_targets)
    keypoint_loss = torch.sum(keypoint_loss, dim=1)
    keypoint_loss = keypoint_loss * masks_for_vote
    non_zero_index = torch.sum(masks_for_vote, (1, 2)).nonzero()
    keypoint_loss = torch.mean(
        torch.sum(keypoint_loss, (1, 2))[non_zero_index] / torch.sum(masks_for_vote, (1, 2))[non_zero_index])
    return keypoint_loss


def vote_orientation_loss(keypoint_offsets, proposals, ori_depths, gt_keypoints, keypoint_matched_idxs, masks_for_vote, gt_labels):
    N, _, Nk, K, H, W = keypoint_offsets.shape
    assert H == W
    gt_axis = []
    vote_offset_targets = []
    voter_labels = []

    for proposals_per_image, gt_kp_in_image, midx, ori_depth, label in zip(proposals, gt_keypoints, keypoint_matched_idxs,
                                                                    ori_depths, gt_labels):
        kp = gt_kp_in_image[midx].view(-1, Nk, K, 1, 1).repeat(1, 1, 1, H, W)
        voter = gt_voters(H, ori_depth, proposals_per_image, config.Unreal_camera_mat)
        gt_offset = kp - voter.unsqueeze(1).repeat(1, Nk, 1, 1, 1)
        gt_axis.append(F.normalize((kp[:, 1] - kp[:, 0]), dim=1))
        vote_offset_targets.append(gt_offset)
        voter_labels.append(label[midx]-1)

    voter_labels = torch.cat(voter_labels, dim=0)
    keypoint_offsets = keypoint_offsets[torch.arange(N), voter_labels]

    votes_targets = torch.cat(vote_offset_targets, dim=0)
    gt_axis = torch.cat(gt_axis, dim=0)
    gt_axis_repeat = gt_axis.unsqueeze(1).repeat(1,2,1,1,1)

    axis_loss = torch.cross((keypoint_offsets - votes_targets), gt_axis_repeat, dim=2)
    axis_loss = torch.norm(axis_loss, dim=2)
    keypoint_loss = torch.mean((keypoint_offsets - votes_targets).abs(), dim=2)
    keypoint_loss = keypoint_loss + axis_loss
    keypoint_loss = torch.mean(keypoint_loss, dim=1)
    axis_direction_loss = 1-torch.sum(F.normalize(keypoint_offsets[:, 1] - keypoint_offsets[:, 0], dim=1)*gt_axis, dim=1)
    keypoint_loss = (keypoint_loss+5*axis_direction_loss)*masks_for_vote
    non_zero_index = torch.sum(masks_for_vote, (1, 2)).nonzero()
    keypoint_loss = torch.mean(
        torch.sum(keypoint_loss, (1, 2))[non_zero_index] / torch.sum(masks_for_vote, (1, 2))[non_zero_index])

    return keypoint_loss

def vote_axis_loss(keypoint_offsets, proposals, ori_depths, gt_keypoints, keypoint_matched_idxs, masks_for_vote, gt_labels):
    N, _, _, H, W = keypoint_offsets.shape
    assert H == W
    gt_axis = []
    vote_offset_targets = []
    voter_labels = []

    for proposals_per_image, gt_kp_in_image, midx, ori_depth, label in zip(proposals, gt_keypoints, keypoint_matched_idxs,
                                                                    ori_depths, gt_labels):
        kp = gt_kp_in_image[midx].view(-1, 2, 3, 1, 1).repeat(1, 1, 1, H, W)
        voter = gt_voters(H, ori_depth, proposals_per_image, config.Unreal_camera_mat)
        gt_offset = kp - voter.unsqueeze(1).repeat(1, 2, 1, 1, 1)
        gt_axis.append(F.normalize((kp[:, 1] - kp[:, 0]), dim=1))
        voter_labels.append(label[midx] - 1)
        vote_offset_targets.append(gt_offset)

    voter_labels = torch.cat(voter_labels, dim=0)
    keypoint_offsets = keypoint_offsets[torch.arange(N), voter_labels]

    votes_targets = torch.cat(vote_offset_targets, dim=0)
    voter_axis_labels = torch.norm(votes_targets, dim=2).argmin(dim=1).float()
    label_loss = F.binary_cross_entropy_with_logits(keypoint_offsets[:, 3], voter_axis_labels)

    gt_axis = torch.cat(gt_axis, dim=0)

    axis_loss = torch.cross((keypoint_offsets[:, :3] - votes_targets[:, 1]), gt_axis, dim=1)
    axis_loss = torch.norm(axis_loss, dim=1)

    norm_loss = torch.abs(torch.sum(keypoint_offsets[:, :3] * gt_axis, dim=1))
    axis_loss = (axis_loss+norm_loss) * masks_for_vote
    non_zero_index = torch.sum(masks_for_vote, (1, 2)).nonzero()
    axis_loss = torch.mean(
        torch.sum(axis_loss, (1, 2))[non_zero_index] / torch.sum(masks_for_vote, (1, 2))[non_zero_index])

    return axis_loss + label_loss

def calculate_norm_vectors(norm_vectors, gt_keypoints, matched_idxs, masks_for_vote, gt_labels):
    N, _, K, H, W = norm_vectors.shape
    assert H == W
    gt_axis = []
    voter_labels = []
    norm_vectors = F.normalize(norm_vectors, dim=2)
    for gt_kp_in_image, midx, label in zip(gt_keypoints, matched_idxs, gt_labels):
        kp = gt_kp_in_image[midx].view(-1, 2, K, 1, 1).repeat(1, 1, 1, H, W)
        gt_axis.append(F.normalize((kp[:, 1] - kp[:, 0]), dim=1))
        voter_labels.append(label[midx] - 1)

    voter_labels = torch.cat(voter_labels, dim=0)
    norm_vectors = norm_vectors[torch.arange(N), voter_labels]

    gt_axis = torch.cat(gt_axis, dim=0)

    norm_loss = torch.mean((norm_vectors-gt_axis).pow(2), dim=1)

    norm_loss = norm_loss * masks_for_vote
    non_zero_index = torch.sum(masks_for_vote, (1, 2)).nonzero()
    norm_loss = torch.mean(
        torch.sum(norm_loss, (1, 2))[non_zero_index] / torch.sum(masks_for_vote, (1, 2))[non_zero_index])

    return norm_loss