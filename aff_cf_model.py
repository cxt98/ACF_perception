import torch
import torch.nn as nn

from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection.rpn import RegionProposalNetwork, AnchorGenerator, RPNHead

from keypoint_3d.kp_3d_roi_heads_attention import RoIHeadsExtend
from collections import OrderedDict
import config

model_urls = {
    'maskrcnn_resnet50_fpn_coco':
        'https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth',
}


class ACFNetwork(nn.Module):
    """Wrapper for pre-built PyTorch models.

    Based off:
        https://pytorch.org/docs/stable/_modules/torchvision/models/detection/mask_rcnn.html
        https://pytorch.org/docs/stable/_modules/torchvision/models/detection/faster_rcnn.html
        https://github.com/pytorch/vision/blob/master/torchvision/models/detection/generalized_rcnn.py
    """

    def __init__(self, arch, pretrained, num_classes, input_mode, acf_head='endpoints',
                 # transform parameters
                 min_size=800, max_size=1333, image_mean=None, image_std=None,
                 # RPN parameters
                 rpn_anchor_generator=None, rpn_head=None,
                 rpn_pre_nms_top_n_train=2000, rpn_pre_nms_top_n_test=1000,
                 rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=1000,
                 rpn_nms_thresh=0.5,
                 rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,
                 rpn_batch_size_per_image=256, rpn_positive_fraction=0.5,
                 # Box parameters
                 box_roi_pool=None, box_head=None, box_predictor=None,
                 box_score_thresh=0.05, box_nms_thresh=0.5, box_detections_per_img=100,
                 box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5,
                 box_batch_size_per_image=512, box_positive_fraction=0.25,
                 bbox_reg_weights=None
                 ):
        super(ACFNetwork, self).__init__()

        self.input_mode = input_mode

        self.backbone = resnet_fpn_backbone(arch, pretrained)
        # change first layer to 4 channel for early fusion with 1 channel depth, load pretrained weights on RGB channels

        conv1_weight_old = nn.Parameter(self.backbone.body.conv1.weight.data)  # self.backbone.body.conv1.weight
        conv1_weight = torch.zeros((64, 4, 7, 7))
        conv1_weight[:, 0:3, :, :] = conv1_weight_old
        avg_weight = conv1_weight_old.mean(dim=1, keepdim=False)
        conv1_weight[:, 3, :, :] = avg_weight
        self.backbone.body.conv1.weight = torch.nn.Parameter(conv1_weight)

        # self.backbone.body.conv1.weight.detach()
        # self.backbone.body.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)

        out_channels = self.backbone.out_channels
        if rpn_anchor_generator is None:
            anchor_sizes = ((16,), (32,), (64,), (128,), (256,), (512,))
            aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
            rpn_anchor_generator = AnchorGenerator(
                anchor_sizes, aspect_ratios
            )
        if rpn_head is None:
            rpn_head = RPNHead(
                out_channels, rpn_anchor_generator.num_anchors_per_location()[0]
            )

        rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)

        self.rpn = RegionProposalNetwork(
            rpn_anchor_generator, rpn_head,
            rpn_fg_iou_thresh, rpn_bg_iou_thresh,
            rpn_batch_size_per_image, rpn_positive_fraction,
            rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_nms_thresh)

        self.roi_heads = RoIHeadsExtend(out_channels, num_classes, self.input_mode, acf_head)

        # freeze RGB backbone and RPN when training on poses
        if self.input_mode == config.INPUT_RGBD:
            for param in self.rpn.parameters():
                param.requires_grad = False
            for param in self.backbone.parameters():
                param.requires_grad = False
            # self.backbone_depth = resnet_fpn_backbone(arch, pretrained)

    def forward(self, images, targets=None):
        """
        Arguments:
            images: Image batch, normalized [NxCxHxW]
            targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)
        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).
        """
        image_sizes = [tuple(images.shape[-2:])] * images.shape[0]

        features = self.backbone(images)

        # Might need to torch.chunk the features because it wants it to be a list for some reason.
        image_list = ImageList(images, image_sizes)
        try:
            proposals, proposal_losses = self.rpn(image_list, features, targets)
        except Exception as e:
            print(e)  # dirty data not cleaned
        detections, detector_losses = self.roi_heads(features, proposals, image_sizes, targets)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        if targets is not None:
            return detections, features, losses
        else:
            return detections, features
