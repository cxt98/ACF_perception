import argparse
from robot import Robot


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='...')
    parser.add_argument('--checkpoint-path', default='./weight', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
    parser.add_argument('--resume', default='endpoints_attention', type=str, help='pth file to use')
    parser.add_argument('--acf-head', default='endpoints', type=str, help='the representation of acf')
    parser.add_argument('-j', '--workers', type=int, default=1)
    parser.add_argument('--out-dir', type=str, default='', help='output image path')
    parser.add_argument('--texture_folder', default=None, type=str, help='texture folder')
    parser.add_argument('--pipeline_id', default=0, type=int, help='0: grasp pipeline: pick every individual objects lying on the tabletop; '
                                                                   '1: pour action: pick a pourer with liquid inside, move above the container and pour liquid into it; '
                                                                   '2: stir action: pick a spoon from tabletop, use that to stir liquid in a container; '
                                                                   '3: whole drink serving task: pick two pourers and pour liquid into a container, then grasp a spoon to stir')
    parser.add_argument('--test_times', default=100, type=int, help='number of repeat tests')
    parser.add_argument('--use_perception', default=True, type=bool, help='True to use ACF perception to estimate ACF keypoints, axes, False to use ground truth instead')
    parser.add_argument('--use_second_camera', default=True, type=bool, help='use 2nd camera to help get better view for perception, False to only use 1 camera, nonsense when use_perception is False')
    parser.add_argument('--experiments_out_dir', default='./vrep/experiments_out', type=str, help='path to save experiment result (scene object configuration + estimation results if perception is enabled)')
    parser.add_argument('--repeat_history_prob', default=0, type=float, help='probability to repeat previous scenes, 1 to load a specific scene (need previous_scene_folder to be set), 0-1 to randomly load from previous scenes in experiments_out_dir')
    parser.add_argument('--repeat_single_object', default=0, type=float, help='probability to repeat a single object, independent for every object')
    parser.add_argument('--previous_scene_folder', default='', help='previous scene folder to repeat when setting repeat_history_prob and repeat_single_object to 1')
    parser.add_argument('--random_texture', default=True, type=bool, help='use randomized textures and colors for all objects with probability')
    args = parser.parse_args()
    robot = Robot('LBR4p', args)
    robot.benchmark_test()
