import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
import config

class ContextBlock(nn.Module):
    def __init__(self,inplanes,ratio,pooling_type='att',
                 fusion_types=('channel_add', )):
        super(ContextBlock, self).__init__()
        valid_fusion_types = ['channel_add', 'channel_mul']

        assert pooling_type in ['avg', 'att']
        assert isinstance(fusion_types, (list, tuple))
        assert all([f in valid_fusion_types for f in fusion_types])
        assert len(fusion_types) > 0, 'at least one fusion should be used'

        self.inplanes = inplanes
        self.ratio = ratio
        self.planes = int(inplanes * ratio)
        self.pooling_type = pooling_type
        self.fusion_types = fusion_types

        if pooling_type == 'att':
            self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if 'channel_add' in fusion_types:
            self.channel_add_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),  # yapf: disable
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusion_types:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),  # yapf: disable
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_mul_conv = None


    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        if self.pooling_type == 'att':
            input_x = x
            # [N, C, H * W]
            input_x = input_x.view(batch, channel, height * width)
            # [N, 1, C, H * W]
            input_x = input_x.unsqueeze(1)
            # [N, 1, H, W]
            context_mask = self.conv_mask(x)
            # [N, 1, H * W]
            context_mask = context_mask.view(batch, 1, height * width)
            # [N, 1, H * W]
            context_mask = self.softmax(context_mask)
            # [N, 1, H * W, 1]
            context_mask = context_mask.unsqueeze(-1)
            # [N, 1, C, 1]
            context = torch.matmul(input_x, context_mask)
            # [N, C, 1, 1]
            context = context.view(batch, channel, 1, 1)
        else:
            # [N, C, 1, 1]
            context = self.avg_pool(x)
        return context

    def forward(self, x):
        # [N, C, 1, 1]
        context = self.spatial_pool(x)
        out = x
        if self.channel_mul_conv is not None:
            # [N, C, 1, 1]
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
            out = out * channel_mul_term
        if self.channel_add_conv is not None:
            # [N, C, 1, 1]
            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term
        return out

class DenseGlobalFeatures(nn.Module):
    def __init__(self, kernel_size):
        super(DenseGlobalFeatures, self).__init__()
        self.conv2_rgbd = torch.nn.Conv1d(256, 512, 1)

        self.conv3 = torch.nn.Conv1d(512, 1024, 1)

        self.ap1 = torch.nn.AvgPool1d(kernel_size)

    def forward(self, emb):
        bs, c, h, w = emb.size()
        x = emb.view(bs, c, -1)
        feat_1 = F.relu(self.conv2_rgbd(x))

        rgbd = F.relu(self.conv3(feat_1))

        ap_x = self.ap1(rgbd)

        ap_x = ap_x.view(-1, 1024, 1).repeat(1, 1, h*w)
        return torch.cat([x, feat_1, ap_x], 1).view(bs, -1, h, w)


class Vote_Kpoints_head(nn.Sequential):
    def __init__(self, in_channels, layers, type="conv1d"):
        d = []
        next_feature = in_channels
        for l in layers:
            if type == "conv2d":
                d.append(nn.Conv2d(next_feature, l, 3, stride=1, padding=1))
            else:
                d.append(nn.Conv1d(next_feature, l, 1))
            d.append(nn.ReLU(inplace=True))
            next_feature = l
        super(Vote_Kpoints_head, self).__init__(*d)
        for m in self.children():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.constant_(m.bias, 0)


class Vote_Kpoints_Predictor(nn.Module):
    def __init__(self, in_channels, num_keypoints):
        super(Vote_Kpoints_Predictor, self).__init__()
        input_features = in_channels
        deconv_kernel = 4
        self.kps_score_lowres = nn.ConvTranspose2d(
            input_features,
            num_keypoints,
            deconv_kernel,
            stride=2,
            padding=deconv_kernel // 2 - 1,
        )
        nn.init.kaiming_normal_(
            self.kps_score_lowres.weight, mode="fan_out", nonlinearity="relu"
        )
        nn.init.constant_(self.kps_score_lowres.bias, 0)
        self.up_scale = 2
        self.out_channels = num_keypoints

    def forward(self, x):
        x = self.kps_score_lowres(x)
        x = F.interpolate(
            x, scale_factor=self.up_scale, mode="bilinear", align_corners=False
        )
        return x

def gaussian_kernel(distance, bandwidth):
    return (1 / (bandwidth * torch.sqrt(2 * torch.tensor(np.pi)))) \
        * torch.exp(-0.5 * ((distance / bandwidth)) ** 2)

class MeanShiftTorch():
    def __init__(self, bandwidth=0.05, max_iter=100):
        self.bandwidth = bandwidth
        self.stop_thresh = bandwidth * 1e-3
        self.max_iter = max_iter

    def fit(self, A):
        """
        params: A: [N, 3]
        """
        N, c = A.size()
        it = 0
        C = A.clone()
        while True:
            it += 1
            Ar = A.view(1, N, c).repeat(N, 1, 1)
            Cr = C.view(N, 1, c).repeat(1, N, 1)
            dis = torch.norm(Cr - Ar, dim=2)
            w = gaussian_kernel(dis, self.bandwidth).view(N, N, 1)
            new_C = torch.sum(w * Ar, dim=1) / torch.sum(w, dim=1)
            # new_C = C + shift_offset
            Adis = torch.norm(new_C - C, dim=1)
            # print(C, new_C)
            C = new_C
            if torch.max(Adis) < self.stop_thresh or it > self.max_iter:
                # print("torch meanshift total iter:", it)
                break
        # find biggest cluster
        Cr = A.view(N, 1, c).repeat(1, N, 1)
        dis = torch.norm(Ar - Cr, dim=2)
        num_in = torch.sum(dis < self.bandwidth, dim=1)
        max_num, max_idx = torch.max(num_in, 0)
        labels = dis[max_idx] < self.bandwidth
        return C[max_idx, :], labels

    def fit_batch_npts(self, A):
        """
        params: A: [bs, n_kps, pts, 3]
        """
        bs, n_kps, N, cn = A.size()
        it = 0
        C = A.clone()
        while True:
            it += 1
            Ar = A.view(bs, n_kps, 1, N, cn).repeat(1, 1, N, 1, 1)
            Cr = C.view(bs, n_kps, N, 1, cn).repeat(1, 1, 1, N, 1)
            dis = torch.norm(Cr - Ar, dim=4)
            w = gaussian_kernel(dis, self.bandwidth).view(bs, n_kps, N, N, 1)
            new_C = torch.sum(w * Ar, dim=3) / torch.sum(w, dim=3)
            # new_C = C + shift_offset
            Adis = torch.norm(new_C - C, dim=3)
            # print(C, new_C)
            C = new_C
            if torch.max(Adis) < self.stop_thresh or it > self.max_iter:
                # print("torch meanshift total iter:", it)
                break
        # find biggest cluster
        Cr = A.view(N, 1, cn).repeat(1, N, 1)
        dis = torch.norm(Ar - Cr, dim=4)
        num_in = torch.sum(dis < self.bandwidth, dim=3)
        # print(num_in.size())
        max_num, max_idx = torch.max(num_in, 2)
        dis = torch.gather(dis, 2, max_idx.reshape(bs, n_kps, 1))
        labels = dis < self.bandwidth
        ctrs = torch.gather(
            C, 2, max_idx.reshape(bs, n_kps, 1, 1).repeat(1, 1, 1, cn)
        )
        return ctrs, labels

def voter_in_proposal(proposal_height, depth_image, bbox, keypoints_offset, camera_matrix):
    bs, _ = bbox.shape
    depth_image = depth_image.squeeze()
    fx, cx, fy, cy = camera_matrix[0, 0], camera_matrix[0, 2], camera_matrix[1, 1], camera_matrix[1, 2]
    offset_x = bbox[:, 0]
    offset_y = bbox[:, 1]

    scale_x = (bbox[:, 2] - bbox[:, 0]) / proposal_height
    scale_y = (bbox[:, 3] - bbox[:, 1]) / proposal_height
    max_x_index = torch.tensor(depth_image.shape[1]-1).view(1, 1, 1).long().cuda()
    max_y_index = torch.tensor(depth_image.shape[0]-1).view(1, 1, 1).long().cuda()
    max_x_index = max_x_index.repeat(bs, proposal_height, proposal_height)
    max_y_index = max_y_index.repeat(bs, proposal_height, proposal_height)

    dst_y, dst_x = torch.meshgrid(torch.arange(proposal_height), torch.arange(proposal_height))
    dst_x = dst_x.view(1, proposal_height, proposal_height).repeat(bs, 1, 1).cuda()
    dst_y = dst_y.view(1, proposal_height, proposal_height).repeat(bs, 1, 1).cuda()
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

def vote_keypoint_inference(ori_depth,keypoint_offsets, box_proposals, masks_for_vote, camera_mat):
    masks_for_vote = F.interpolate(masks_for_vote, scale_factor=2, mode="bilinear", align_corners=False)
    proposal_height = keypoint_offsets.shape[3]
    voters = voter_in_proposal(proposal_height, ori_depth, box_proposals, keypoint_offsets, camera_mat)
    voters = voters + keypoint_offsets
    nozero_index = masks_for_vote > 0.6
    mst = MeanShiftTorch(1)
    keypoints = []
    vaild_voters = []
    for i in range(voters.shape[0]):
        _, index = torch.sort(masks_for_vote[i].view(1, -1), dim=-1)
        index = index.squeeze()[-1000:]
        vaild_voter = voters[i].view(3, -1)[:, index].squeeze().transpose(0, 1)
        ctr, _ = mst.fit(vaild_voter)
        keypoints.append(ctr)
        vaild_voters.append(vaild_voter.cpu().numpy())

    return torch.stack(keypoints, dim=0), vaild_voters

def refine_vote_axis(voters, axis_keypoints):
    from skimage.measure import LineModelND, ransac

    voters = np.array(voters).transpose([1,0,2,3]).reshape((-1, 2000, 3))
    axis = []
    for i in range(voters.shape[0]):
        model_robust, _ = ransac(voters[i], LineModelND,
                                       min_samples=10,
                                       residual_threshold=1, max_trials=1000)
        axis.append(model_robust.params[1])
    regress_axis = torch.from_numpy(np.array(axis))
    error = (axis_keypoints*regress_axis).sum(1)
    threshold = 0.9
    error[error>threshold] = 1
    error[error<-threshold] = -1
    trust_regress_index = error*(error.abs()>threshold).float()
    refine_axis = axis_keypoints * (error.abs() < threshold).unsqueeze(1).float() + regress_axis * trust_regress_index.unsqueeze(1)
    return  refine_axis

def vote_axis_inference(ori_depth,keypoint_offsets, box_proposals, masks_for_vote, labels):
    from skimage.measure import LineModelND, ransac

    masks_for_vote = F.interpolate(masks_for_vote, scale_factor=2, mode="bilinear", align_corners=False)
    proposal_height = keypoint_offsets.shape[3]
    voters = voter_in_proposal(proposal_height, ori_depth, box_proposals, keypoint_offsets, config.Unreal_camera_mat)
    ori_voters = voters
    voters = voters + keypoint_offsets[:, :3]
    voter_label = keypoint_offsets[:, 3].sigmoid() > 0.5
    voters = torch.cat((voters, voter_label.float().unsqueeze(dim=1)), dim=1)
    # nozero_index = masks_for_vote > 0.6
    axis = []
    vaild_voters = []
    vaild_ori_voters = []
    for i in range(voters.shape[0]):

        _, index = torch.sort(masks_for_vote[i].view(1, -1), dim=-1)
        index = index.squeeze()[-1000:]
        ori_voter = ori_voters[i].view(3, -1)[:, index].squeeze().transpose(0, 1).cpu()
        vaild_voter = voters[i].view(4, -1)[:, index].squeeze().transpose(0, 1).cpu()
        model_robust, inliers = ransac(vaild_voter[:, :3].numpy(), LineModelND,
                                       min_samples=4,
                                       residual_threshold=1, max_trials=1000)
        in_vaild_voter = vaild_voter[torch.from_numpy(inliers)]
        out_vaild_voter = vaild_voter[torch.from_numpy(inliers==0)]
        line = model_robust.params[1]

        try:
            voters1_index = vaild_voter[:, 3] == 1
            voters1 = vaild_voter[voters1_index, :3]
            voters2 = vaild_voter[1-voters1_index, :3]
            voters1_sample = np.random.choice(range(voters1.shape[0]), size=100, replace=True)
            voters2_sample = np.random.choice(range(voters2.shape[0]), size=100, replace=True)
            voters1 = voters1[voters1_sample]
            voters2 = voters2[voters2_sample]
            vectors = F.normalize((voters1 - voters2),dim=1)
            direction = np.sum(np.dot(vectors.numpy(), line)>0)
            if direction > 50:
                line = line
            else:
                line = -line
        except:
            print('Do not find points for instance{}'.format(i))
        vaild_voters.append(in_vaild_voter)
        vaild_ori_voters.append(ori_voter)
        axis.append(line)

    return torch.from_numpy(np.array(axis)), vaild_voters, vaild_ori_voters