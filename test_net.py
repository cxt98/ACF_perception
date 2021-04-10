import os
import torch
import torch.multiprocessing as mp
import cv2

import aff_cf_model
import acfutils.common_utils as utils
import config


def test_net(args):
    if args.workers > 1:
        mp.set_start_method('spawn')
    import Dataloader as new_datasetload
    test_loader, classes = new_datasetload.make_data_loader('test', args.dataset, args.data_dir, batch_size=args.batch_size,
                                                    workers=args.workers, shuffle=False)
    model = aff_cf_model.ACFNetwork(arch='resnet50', pretrained=True, num_classes=len(classes),
                                    input_mode=args.input_mode, acf_head=args.acf_head)
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    if args.checkpoint_path != '':
        state_dict = torch.load(os.path.join(args.checkpoint_path, args.resume+'.pth'), map_location = device)
        model.load_state_dict(state_dict['model_state_dict'])

    model.to(device)
    model.eval()
    out_imgdir = args.out_dir + '/result_images/'
    out_evldir = args.out_dir + '/result_eval/'
    os.system('mkdir -p ' + out_imgdir)
    os.system('mkdir -p ' + out_evldir)
    os.system('rm '+out_imgdir+'*')

    print('testing output to: ' + out_imgdir)
    proposals_all, targets_all, image_paths_all = [], [], []
    if args.dataset == 'unreal_parts':
        for images, targets in test_loader:
            images = images.to(device)
            with torch.no_grad():
                detections, _ = model(images)
            img_paths = [t['img_file'] for t in targets]
            if detections == []:
                print("Donot find any part in {}".format(img_paths))
                continue
            img_depths = cv2.imread(img_paths[0].replace('.png', '.depth.mm.16.png'), -1)
            proposals = utils.post_process_proposals(detections, img_depths, K=config.BOX_POSTPROCESS_NUMBER)
            final_pafs_pair = None
            try:
                if proposals[0]["paf_vectors"] is not None:
                    final_pafs_pair = utils.pafprocess(proposals, config.Unreal_camera_mat)
                utils.vis_images(proposals, targets, img_paths, classes, config.Unreal_camera_mat, training=False, output_dir=out_imgdir, final_pafs_pair = final_pafs_pair)
                if args.input_mode == 'RGBD':
                    for proposal in proposals:
                        proposals_all.append([proposal[a].cpu().numpy() for a in ['labels', 'keypoints_3d', 'boxes','axis']])
                    for target in targets:
                        target_info = [target[a].cpu().numpy() for a in ['frame', 'labels', 'axis_keypoints']]
                        target_info.append(target['names'])
                        targets_all.append(target_info)
            except Exception as e:
                print(e)
                continue
            # torch.cuda.empty_cache()

        if args.input_mode == 'RGBD':
            results_stats = utils.evaluate(proposals_all, targets_all, classes, out_evldir)
            utils.plot_auc(results_stats, out_evldir)


if __name__ == '__main__':
    args = utils.parse_args('test')
    test_net(args)
