import random
import math
from lib import sim
import time
import os
import torch
import torchvision.transforms.functional as TF
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
import aff_cf_model
from acfutils import common_utils as acfutils
import config
from copy import deepcopy
import pickle
import datetime
import glob


def euler2rotm(euler):
    return R.from_euler('zyx', [euler[2], euler[1], euler[0]]).as_matrix()


def rotm2euler(rotm):
    return R.from_matrix(rotm).as_euler('zyx')[::-1]


def pour_pose_acf(is_bottle, part1_acf, part2_acf, grasp_pose, n_sample=8):
    # part1 should be container body, part2 should be pourer body
    if is_bottle:
        radius_offset = 0.15  # offset from object body keypoint to opening
        lift_distance = 0.2
    else:
        radius_offset = 0
        lift_distance = 0.25
    xy_vector_norm = np.linalg.norm(grasp_pose[0:2, 0])
    pose = np.eye(4)
    pose[0, 3] = part1_acf[0][0] - grasp_pose[0, 0] * radius_offset / xy_vector_norm
    pose[1, 3] = part1_acf[0][1] - grasp_pose[1, 0] * radius_offset / xy_vector_norm
    pose[2, 3] = part1_acf[0][2] + lift_distance
    pose[:3, 2] = part1_acf[1]
    pose[:3, 1] = np.cross([1, 0, 0], pose[:3, 2])
    pose[:3, 1] = pose[:3, 1] / np.linalg.norm(pose[:3, 1])
    pose[:3, 0] = np.cross(pose[:3, 1], pose[:3, 2])
    current_pose = deepcopy(pose)
    current_pose[:3, 3] = part2_acf[0]
    current_pose[:3, 2] = part2_acf[1]
    current_pose[:3, 1] = np.cross([1, 0, 0], current_pose[:3, 2])
    current_pose[:3, 1] = current_pose[:3, 1] / np.linalg.norm(current_pose[:3, 1])
    current_pose[:3, 0] = np.cross(current_pose[:3, 1], current_pose[:3, 2])
    trans2grasp = np.dot(np.linalg.inv(current_pose), grasp_pose)
    pose_list = [pose]
    if n_sample > 1:
        rotate_interval = math.pi * 2 / n_sample
        for i in range(1, n_sample):
            rot_angle = rotate_interval * i
            xy_vector = np.dot([[np.cos(rot_angle), -np.sin(rot_angle)], [np.sin(rot_angle), np.cos(rot_angle)]],
                               grasp_pose[0:2, 0])
            pose1 = deepcopy(pose)
            pose1[0, 3] = part1_acf[0][0] - xy_vector[0] * radius_offset / xy_vector_norm
            pose1[1, 3] = part1_acf[0][1] - xy_vector[1] * radius_offset / xy_vector_norm
            pose1[:3, :3] = np.dot(pose[:3, :3], [[np.cos(rot_angle), -np.sin(rot_angle), 0],
                                                  [np.sin(rot_angle), np.cos(rot_angle), 0], [0, 0, 1]])
            pose_list.append(pose1)
    return [np.dot(pose, trans2grasp) for pose in pose_list]


def stir_pose_acf(part1_acf, part2_acf, part3_acf, grasp_pose, lift_distance=0.2, n_sample=8):
    # part1 should be container body, part2 should be the part to stir (spoon_head), part3 gives another constraints in axis (spoon_stir)
    part2_pose = np.eye(4)
    part2_pose[:3, 3] = part2_acf[0]
    part2_pose[:3, 2] = part2_acf[1]
    part2_pose[:3, 0] = np.cross(part3_acf[1], part2_acf[1])
    part2_pose[:3, 0] = part2_pose[:3, 0] / np.linalg.norm(part2_pose[:3, 0])
    part2_pose[:3, 1] = np.cross(part2_pose[:3, 2], part2_pose[:3, 0])
    trans2grasp = np.dot(np.linalg.inv(part2_pose), grasp_pose)
    pose = np.eye(4)
    pose[:3, 3] = part1_acf[0]
    pose[2, 3] += lift_distance
    pose[:3, 1] = -part1_acf[1]  # have y axis downwards (opposite to body axis)
    pose[:3, 0] = [1, 0, 0]
    pose[:3, 2] = np.cross(pose[:3, 0], pose[:3, 1])
    pose_list = [pose]
    if n_sample > 1:
        rotate_interval = math.pi * 2 / n_sample
        for i in range(1, n_sample):
            rot_angle = rotate_interval * i
            pose1 = deepcopy(pose)
            pose1[:3, :3] = np.dot(pose1[:3, :3], [[np.cos(rot_angle), 0, np.sin(rot_angle)], [0, 1, 0],
                                                   [-np.sin(rot_angle), 0, np.cos(rot_angle)]])
            pose_list.append(pose1)
    return [np.dot(pose, trans2grasp) for pose in pose_list], trans2grasp


def save_error(save_path, error1, error2, error_type):
    txt = open(save_path, 'a+')
    if error1 == [] or error2 == []:
        txt.write('None, None, False, undetected, errors: {} \n'.format(error1 + error2))
    else:
        error = np.array(error1 + error2)
        distance_error = error[:, 0].mean()
        angle_error = error[:, 1].mean()
        succ = True if error_type == 0 else False
        if 'stir' in save_path:
            stir_index = np.where(error[:, -1] == 3)[0]
            if stir_index.size != 0:
                txt.write('{:.3f}, {:.3f}, {}, {}, spoon handle error: {}, {}\n'.format(distance_error, angle_error,
                                                                                        succ, error_type,
                                                                                        error[stir_index[0], 0],
                                                                                        error[stir_index[0], 1]))
            else:
                txt.write('{:.3f}, {:.3f}, {}, {} \n'.format(distance_error, angle_error, succ, error_type))
        else:
            txt.write('{:.3f}, {:.3f}, {}, {} \n'.format(distance_error, angle_error, succ, error_type))
    txt.close()


class Robot(object):
    def __init__(self, base_name, args):
        sim.simxFinish(-1)  # Just in case, close all opened connections
        self.sim_client = sim.simxStart('127.0.0.1', 19997, True, True, -10000, 5)  # Connect to V-REP on port 19997
        if self.sim_client == -1:
            print('Failed to connect to simulation (V-REP remote API server). Exiting.')
            exit()
        else:
            print('Connected to simulation.')
            sim.simxStartSimulation(self.sim_client, sim.simx_opmode_blocking)
            sim_ret, self.robot_handle = sim.simxGetObjectHandle(self.sim_client, base_name,
                                                                 sim.simx_opmode_blocking)

        self.args = args
        self.object_lib = {}
        self.object_list_ = []
        self.texture_imgs = []
        if self.args.texture_folder is not None:
            for file in os.listdir(self.args.texture_folder):
                if file.endswith('.png') or file.endswith('.jpg'):
                    self.texture_imgs.append(file)

        self.empty_buff = bytearray()
        self.particle_objects = []
        self.classes = {'__background__': 0,
                        'body': 1,
                        'handle': 2,
                        'stir': 3,
                        'head': 4}

        self.tabletop_z = 0.5393 - 0.072 / 2  # z, from tap on the table
        self.reference_frame_offset = 0.0222  # a 3D frame for vis
        # Mug1, Bottle4 are in training set
        # Mug5 has problem in grasp checking (always say not grasped, add sensor as another source for grasp checking)
        # bottle5 seems to have collision problem, cannot grasp, 'Bottle5_model'
        # spoon9 seems to have collision problem, will be penetrated by gripper
        # bowl5 cannot hold particles well
        # mug7 seems to have grasp checking problem
        self.active_objects = [
            'Mug2_model', 'Mug3_model', 'Mug4_model', 'Mug5_model', 'Mug6_model',
            'Mug7_model', 'Mug8_model', 'Mug9_model',
            'Bottle1_model', 'Bottle2_model', 'Bottle3_model', 'Bottle6_model', 'Bottle7_model', 'Bottle8_model',
            'Bowl1_model', 'Bowl2_model', 'Bowl3_model', 'Bowl4_model', 'Bowl5_model', 'Bowl6_model', 'Bowl7_model',
            'Bowl8_model', 'Spoon6_model', 'Spoon7_model', 'Spoon8_model', 'Spoon9_model',
            'Spoon10_model', 'Spoon11_model', 'Spoon12_model', 'Spoon13_model'
        ]
        self.object_sets = {
            'pourer': [
                'Mug2_model', 'Mug3_model', 'Mug7_model', 'Bottle1_model'],
            'container': [
                'Bowl4_model', 'Bowl6_model', 'Bowl7_model',
                'Bowl8_model', 'Mug4_model', 'Mug8_model', 'Mug6_model', 'Bottle8_model'],
            'spoon': ['Spoon10_model', 'Spoon11_model', 'Spoon12_model', 'Spoon13_model'],
            'container_stir': [
                'Bowl4_model',
                'Bowl6_model', 'Bowl7_model', 'Bowl8_model', 'Bottle8_model']

        }
        self.read_object_geometry()  # change read_from_file to False to refresh saved file
        self.init_arm_joints = self.read_arm_joints()
        self.path_record = []  # for arm return to default configuration
        self.cam_intrinsic = np.array([[640, 0, 640], [0, 640, 480], [0, 0, 1]])
        if args.use_perception:
            self.set_acf_network(args)
            self.cam_handle, self.cam_pose, self.cam_intrinsic = self.setup_camera()
            if args.use_second_camera:
                self.cam_handle2, self.cam_pose2, self.cam_intrinsic2 = self.setup_camera('Vision_sensor0',
                                                                                          revert_orientation=False)
        self.particle_grid_size = [4, 4, 5]
        self.test_name = ['grasp', 'pour', 'stir', 'drinkserve']
        self.args.current_time = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')
        self.stir_dis_threshold = 0.05
        self.stir_height_offset = 0.05
        self.pour_ratio_threshold = 0.7
        random.seed(datetime.datetime.now())

    def __del__(self):
        # Stop simulation:
        sim.simxStopSimulation(self.sim_client, sim.simx_opmode_oneshot_wait)

        # Now close the connection to V-REP:
        sim.simxFinish(self.sim_client)

    def benchmark_test(self):
        if not os.path.exists(self.args.experiments_out_dir):
            os.makedirs(self.args.experiments_out_dir)
        test_name = self.test_name[self.args.pipeline_id]
        estimate_pose, error_dict = None, None
        for loop in range(self.args.test_times):
            output_dir = os.path.join(self.args.experiments_out_dir,
                                      self.args.current_time + '_{}/{}'.format(test_name, loop))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            scene_save_name = os.path.join(output_dir, test_name + '_test.p')
            scene_created = False
            if np.random.random() < self.args.repeat_history_prob:
                scenename = self.args.previous_scene_folder + test_name + '_test.p'
                try:
                    self.load_previous_scene(test_name, save_scene_name=scene_save_name)
                    scene_created = True
                except FileNotFoundError:
                    if self.args.repeat_history_prob < 1:
                        print('not enough previous scenes')
                    else:
                        print('cannot load from ' + scenename, ', create new scene')
            succ = None
            if test_name == 'grasp':
                if not scene_created:
                    shape_names = self.generate_random_objects(
                        n_objs={'spoon': [1, 3], 'bottle': [1, 3], 'mug': [1, 3], 'bowl': [0, 0]})
                    self.create_scene(shape_names, x_limit=[1, 1.5], y_limit=[-1.8, -1.1],
                                      save_scene_name=scene_save_name)
                i = 0
                obj_list = deepcopy(self.object_list_)
                while i < len(obj_list):
                    i += 1
                    if self.args.use_perception:
                        estimate_pose, error_dict = self.perception_results(out_dir=output_dir,
                                                                            image_save_name='grasping_{}'.format(i))
                    self.grasp_pipeline(estimate_pose, error_dict, obj_list[len(obj_list) - 1 - i],
                                        out_dir=self.args.experiments_out_dir)
                self.clear_workspace()
            elif test_name == 'pour':
                if not scene_created:
                    container_name = self.generate_random_objects_in_set(obj_set_name='container')
                    pourer_name = self.generate_random_objects_in_set(obj_set_name='pourer')
                    self.create_scene(container_name, stand_prob=1.0, x_limit=[1.1, 1.3], y_limit=[-1.6, -1.4],
                                      limited_try=False, check_grasp=False, add_force_sensor=True, z_offset=0.03)
                    self.create_scene(pourer_name, stand_prob=1.0, x_limit=[1, 1.5], y_limit=[-1.8, -1.1],
                                      limited_try=False, check_distance=True, save_scene_name=scene_save_name,
                                      add_force_sensor=True, z_offset=0.03)
                if self.args.use_perception:
                    estimate_pose, error_dict = self.perception_results(out_dir=output_dir, image_save_name='pouring')
                container = self.object_list_[0]
                pourer = self.object_list_[1]
                succ = self.pour_pipeline(container, pourer, estimate_pose)
                self.clear_workspace()
                if self.args.use_perception:
                    save_error(os.path.join(self.args.experiments_out_dir, 'pour_pipeline.txt'),
                                    error_dict[container['handle']], error_dict[pourer['handle']], succ)
            elif test_name == 'stir':
                if not scene_created:
                    container_name = self.generate_random_objects_in_set(obj_set_name='container_stir')
                    spoon_name = self.generate_random_objects_in_set(obj_set_name='spoon')
                    self.create_scene(container_name, stand_prob=1.0, x_limit=[1.15, 1.25], y_limit=[-1.55, -1.45],
                                      limited_try=False, check_grasp=False)
                    self.create_scene(spoon_name, x_limit=[0.95, 1.45], y_limit=[-1.75, -1.25], limited_try=False,
                                      save_scene_name=scene_save_name, add_convex=True, check_distance=True)
                if self.args.use_perception:
                    estimate_pose, error_dict = self.perception_results(out_dir=output_dir, image_save_name='stirring')
                container = self.object_list_[0]
                spoon = self.object_list_[1]
                if estimate_pose[container['handle']] != [] and estimate_pose[spoon['handle']] != []:
                    succ = self.stir_pipeline(container, spoon, estimate_pose)
                else:
                    print('cannot detect spoon or container')
                self.clear_workspace()
                if self.args.use_perception:
                    save_error(os.path.join(self.args.experiments_out_dir, 'stir_pipeline.txt'),
                                    error_dict[container['handle']], error_dict[spoon['handle']], succ)
            elif test_name == 'drinkserve':
                if not scene_created:
                    container_name = self.generate_random_objects_in_set(obj_set_name='container_stir')
                    pourer_name = self.generate_random_objects_in_set(obj_set_name='pourer', n_objs=2)
                    spoon_name = self.generate_random_objects_in_set(obj_set_name='spoon')
                    self.create_scene(container_name, stand_prob=1.0, x_limit=[1.15, 1.25], y_limit=[-1.55, -1.45],
                                      limited_try=False, check_grasp=False, add_force_sensor=True, z_offset=0.03)
                    self.create_scene(spoon_name, x_limit=[0.95, 1.45], y_limit=[-1.75, -1.25], limited_try=False,
                                      check_distance=True, add_convex=True)
                    self.create_scene(pourer_name, stand_prob=1.0, x_limit=[0.95, 1.55], y_limit=[-1.8, -1.1],
                                      limited_try=False, check_distance=True, save_scene_name=scene_save_name,
                                      add_force_sensor=True, z_offset=0.03)
                if self.args.use_perception:
                    estimate_pose, error_dict = self.perception_results(out_dir=output_dir,
                                                                        image_save_name='drinkserving')
                self.drinkserve_pipeline(self.object_list_[0], self.object_list_[1], self.object_list_[2:],
                                         estimate_pose, error_dict, error_save_dir=self.args.experiments_out_dir)
                self.clear_workspace()
            else:
                print('wrong pipeline id, valid: 0-3')
                break

    # # setup functions
    def read_object_geometry(self, read_from_file=True, save_filename='object_geom.p'):
        if read_from_file:
            try:
                print('load saved object geometry {}'.format(save_filename))
                self.object_lib = pickle.load(open(save_filename, 'rb'))
                return
            except:
                print('cannot load from object geometry file, read from simulator instead')

        self.object_lib['object_n'] = {}
        self.object_lib['names'] = {}
        n_mug, n_bottle, n_spoon, n_bowl = 0, 0, 0, 0
        names_mug, names_bottle, names_spoon, names_bowl = [], [], [], []
        for obj in self.active_objects:
            self.object_lib[obj] = {}
            _, object_handle = sim.simxGetObjectHandle(self.sim_client, obj, sim.simx_opmode_blocking)
            _, xmin = sim.simxGetObjectFloatParameter(self.sim_client, object_handle, 15, sim.simx_opmode_blocking)
            _, ymin = sim.simxGetObjectFloatParameter(self.sim_client, object_handle, 16, sim.simx_opmode_blocking)
            _, zmin = sim.simxGetObjectFloatParameter(self.sim_client, object_handle, 17, sim.simx_opmode_blocking)
            _, xmax = sim.simxGetObjectFloatParameter(self.sim_client, object_handle, 18, sim.simx_opmode_blocking)
            _, ymax = sim.simxGetObjectFloatParameter(self.sim_client, object_handle, 19, sim.simx_opmode_blocking)
            _, zmax = sim.simxGetObjectFloatParameter(self.sim_client, object_handle, 20, sim.simx_opmode_blocking)
            self.object_lib[obj]['size'] = [xmax - xmin, ymax - ymin, zmax - zmin]
            name_prefix = obj[:-5]  # need careful naming in Coppeliasim: 'ObjectName_model'
            if 'Mug' in obj:
                parts = ['body', 'handle']
                n_mug += 1
                names_mug.append(obj)
            elif 'Bottle' in obj or 'Bowl' in obj:
                parts = ['body']
                if 'Bottle' in obj:
                    n_bottle += 1
                    names_bottle.append(obj)
                else:
                    n_bowl += 1
                    names_bowl.append(obj)
            elif 'Spoon' in obj:
                parts = ['stir', 'head']
                n_spoon += 1
                names_spoon.append(obj)
            for part in parts:
                _, dummy_handle = sim.simxGetObjectHandle(self.sim_client, 'Dummy_' + name_prefix + part,
                                                          sim.simx_opmode_blocking)
                _, keypoint_pos = sim.simxGetObjectPosition(self.sim_client, dummy_handle, object_handle,
                                                            sim.simx_opmode_blocking)
                _, axis_euler = sim.simxGetObjectOrientation(self.sim_client, dummy_handle, object_handle,
                                                             sim.simx_opmode_blocking)  # https://www.coppeliarobotics.com/helpFiles/en/eulerAngles.htm
                axis = euler2rotm(axis_euler)[:, 2]  # always z axis
                self.object_lib[obj][part] = [keypoint_pos, axis]
        self.object_lib['object_n']['mug'] = n_mug
        self.object_lib['object_n']['bottle'] = n_bottle
        self.object_lib['object_n']['bowl'] = n_bowl
        self.object_lib['object_n']['spoon'] = n_spoon
        self.object_lib['names']['mug'] = names_mug
        self.object_lib['names']['bottle'] = names_bottle
        self.object_lib['names']['bowl'] = names_bowl
        self.object_lib['names']['spoon'] = names_spoon

        pickle.dump(self.object_lib, open(save_filename, 'wb'))

    def setup_camera(self, camera_name='Vision_sensor', revert_orientation=True):
        sim_ret, cam_handle = sim.simxGetObjectHandle(self.sim_client, camera_name, sim.simx_opmode_blocking)
        sim_ret, cam_position = sim.simxGetObjectPosition(self.sim_client, cam_handle, -1, sim.simx_opmode_blocking)
        sim_ret, cam_orientation = sim.simxGetObjectOrientation(self.sim_client, cam_handle, -1,
                                                                sim.simx_opmode_blocking)
        cam_trans = np.eye(4, 4)
        cam_trans[0:3, 3] = np.asarray(cam_position)
        if revert_orientation:
            cam_orientation = [-cam_orientation[0], -cam_orientation[1], -cam_orientation[2]]  # why negative?
        cam_rotm = np.eye(4, 4)
        r = R.from_euler('xyz', cam_orientation)
        cam_rotm[:3, :3] = r.as_matrix()
        cam_rotm[:, :2] = -cam_rotm[:, :2]  # negate x and y axes, found from example
        cam_pose = np.dot(cam_trans, cam_rotm)  # Compute rigid transformation representating camera pose
        cam_intrinsic = self.cam_intrinsic
        return cam_handle, cam_pose, cam_intrinsic

    def load_previous_scene(self, test_name, save_scene_name=None):
        # if args.repeat_history_prob < 1, randomly select a previous scene and randomly repeat or create new same object in the scene
        if self.args.repeat_history_prob < 1:
            previous_scene_list = []
            for f in glob.glob(self.args.experiments_out_dir + '/**', recursive=True):
                if f.endswith(test_name + '_test.p'):
                    previous_scene_list.append(f)
            if len(previous_scene_list) < 10:
                raise FileNotFoundError
            random_scene_name = random.sample(previous_scene_list, 1)[0]
            print('randomly select scene ' + random_scene_name)
            read_object_list = pickle.load(open(random_scene_name, 'rb'))
        else:
            load_scene_name = self.args.previous_scene_folder + test_name + '_test.p'
            print('load scene ' + load_scene_name)
            read_object_list = pickle.load(open(load_scene_name, 'rb'))

        random_objects = []
        for ob in read_object_list:
            texture_path = ''
            if self.args.random_texture:
                texture_path = self.args.texture_folder + '/' + self.texture_imgs[
                    np.random.randint(len(self.texture_imgs))]
            add_convex = False
            if 'Spoon' in ob['name']:
                add_convex = True
            if np.random.random() < self.args.repeat_single_object:
                print('Copy {} from the saved scene'.format(ob['name']))
                _, ret_ints, _, _, _ = self.call_remote('importShape_new',
                                                        int_param=[len(self.object_list_) + 1, self.args.random_texture,
                                                                   ob['forcesensor'] != -1, add_convex],
                                                        float_param=ob['pos'] + ob['ori'] + [
                                                            self.object_lib[ob['name']]['size'][2] / 2],
                                                        str_param=[ob['name'], texture_path])
                ob['forcesensor'] = ret_ints[1]
                if add_convex:
                    ob['handle'], ob['convex_handle'] = ret_ints[2], ret_ints[0]
                else:
                    ob['handle'], ob['convex_handle'] = ret_ints[0], ret_ints[2]
                self.object_list_.append(ob)
                self.update_collision_set()
            else:
                random_objects.append(ob)

        for ob in random_objects:
            self.create_scene([ob['name']], ob['stand_prob'], ob['x_limit'], ob['y_limit'], ob['z_offset'],
                              ob['limited_try'], ob['check_distance'], ob['check_grasp'], None, ob['forcesensor'] != -1)

        if self.args.repeat_history_prob < 1:
            pickle.dump(self.object_list_, open(save_scene_name, 'wb'))

    def create_scene(self, shape_names, stand_prob=0.7, x_limit=None, y_limit=None, z_offset=0.001, limited_try=True,
                     check_distance=False, check_grasp=True, save_scene_name=None, add_force_sensor=False,
                     add_convex=False):
        """
        :param add_convex: add convex hull of object, to stablize simulation during grasping, currently only used for spoons
        :param add_force_sensor: force sensor to measure weight of object (with particles) to evaluate pour action
        :param save_scene_name: save scene name (always save after generating a scene)
        :param shape_names: object names to generate
        :param stand_prob: probability of generating objects that stand on tabletop
        :param limited_try: whether try limited times for each object
        :param check_distance: whether check near distance between every two objects during generation
        :param check_grasp: whether check available pre-grasp and grasp pose during generation
        :return:
        """
        print('create new scene')
        if y_limit is None:
            y_limit = [-1.7, -1.2]
        if x_limit is None:
            x_limit = [1.2, 1.5]
        x_min, y_min, x_max, y_max = x_limit[0], y_limit[0], x_limit[1], y_limit[1]
        pregrasp_step = 0.05

        for shape_name in shape_names:
            print('try to create {}'.format(shape_name))
            texture_path = ''
            if self.args.random_texture:
                texture_path = self.args.texture_folder + '/' + self.texture_imgs[
                    np.random.randint(len(self.texture_imgs))]
            check_passed = False
            counter = 0
            while not check_passed:
                if np.random.random() > stand_prob and 'Spoon' not in shape_name:  # random laying down pose, only has rotation in horizontal plane, [90, x, 0], center = tabletop + y / 2
                    object_position = [np.random.uniform(x_min, x_max),
                                       np.random.uniform(y_min, y_max),
                                       self.tabletop_z + self.object_lib[shape_name]['size'][1] / 2 + z_offset]
                    object_orientation = [-math.pi / 2, np.random.uniform(-math.pi, math.pi), 0]
                else:  # random standing pose, [90, x, 90], center = tabletop + z / 2
                    object_position = [np.random.uniform(x_min, x_max),
                                       np.random.uniform(y_min, y_max),
                                       self.tabletop_z + self.object_lib[shape_name]['size'][2] / 2 + z_offset]
                    object_orientation = [0, 0, np.random.uniform(-math.pi, math.pi)]

                _, ret_ints, _, _, _ = self.call_remote('importShape_new',
                                                        int_param=[1 + len(self.object_list_), self.args.random_texture,
                                                                   add_force_sensor, add_convex],
                                                        float_param=object_position + object_orientation + [
                                                            self.object_lib[shape_name]['size'][2] / 2],
                                                        str_param=[shape_name, texture_path])

                obj = {'name': shape_name, 'pos': object_position, 'ori': object_orientation,
                       'forcesensor': ret_ints[1], 'x_limit': x_limit, 'y_limit': y_limit, 'z_offset': z_offset,
                       'limited_try': limited_try, 'check_distance': check_distance, 'check_grasp': check_grasp,
                       'stand_prob': stand_prob}
                if add_convex:
                    obj['handle'], obj['convex_handle'] = ret_ints[2], ret_ints[0]
                else:
                    obj['handle'], obj['convex_handle'] = ret_ints[0], ret_ints[2]
                self.object_list_.append(obj)
                self.update_collision_set()
                time.sleep(1)
                if add_convex:
                    _, iscollision, _, _, _ = self.call_remote('check_collision', int_param=[ret_ints[2]])
                else:
                    _, iscollision, _, _, _ = self.call_remote('check_collision', int_param=[ret_ints[0]])
                if iscollision[0]:
                    self.remove_object(obj)
                    continue
                check_passed = True
                if self.args.use_perception:
                    estimate_pose, _ = self.perception_results()
                    if self.args.pipeline_id:
                        for id in estimate_pose:
                            if estimate_pose[id] == []:
                                check_passed = False
                                print('fail to detected in create scene')
                                break
                        if not check_passed:
                            self.remove_object(obj)
                            continue
                    else:
                        if list(estimate_pose.items())[-1][1] == []:
                            check_passed = False
                            print('fail to detected in create scene')
                            self.remove_object(obj)
                            continue
                if check_distance:
                    for prev_obj in self.object_list_[:-1]:
                        prev_obj_size, obj_size = self.object_lib[prev_obj['name']]['size'], \
                                                  self.object_lib[obj['name']]['size']
                        prev_obj_pos, obj_pos = np.array(prev_obj['pos']), np.array(obj['pos'])
                        if np.linalg.norm(prev_obj_pos[:2] - obj_pos[:2]) < (
                                np.linalg.norm(obj_size[:2]) + np.linalg.norm(prev_obj_size[:2])) / 2 * 0.8:
                            # compare center distance to diagonal length sum in horizontal plane
                            check_passed = False
                            break
                    if not check_passed:
                        self.remove_object(obj)
                        continue
                if check_grasp:
                    counter += 1
                    trans = np.eye(4, 4)
                    trans[0:3, 3] = np.asarray(object_position)
                    rotm = np.eye(4, 4)
                    rotm[:3, :3] = euler2rotm(object_orientation)
                    obj_pose = np.dot(trans, rotm)
                    grasp_pose_list = self.calculate_grasp_pose(obj, obj_pose)
                    for grasp_pose in grasp_pose_list:
                        offset_mat = np.zeros((4, 4))
                        offset_mat[:3, -1] = pregrasp_step * grasp_pose[:3, 2]
                        pre_grasp_pose = grasp_pose - offset_mat
                        check_passed, path = self.plan(pre_grasp_pose)
                        if check_passed:
                            break
                    if not check_passed:
                        self.remove_object(obj)
                    if limited_try and counter >= 5:
                        break
        if save_scene_name is not None:
            pickle.dump(self.object_list_, open(save_scene_name, 'wb'))

    # # perception interface
    def get_camera_data(self, cam_separate_handle):
        # Get color image from simulation
        sim_ret, resolution, raw_image = sim.simxGetVisionSensorImage(self.sim_client, cam_separate_handle, 0,
                                                                      sim.simx_opmode_blocking)
        color_img = np.asarray(raw_image)
        color_img.shape = (resolution[1], resolution[0], 3)
        color_img = color_img.astype(np.float) / 255
        color_img[color_img < 0] += 1
        color_img *= 255
        # color_img = np.fliplr(color_img)
        color_img = np.flipud(color_img)
        color_img = color_img.astype(np.uint8)

        # Get depth image from simulation
        sim_ret, resolution, depth_buffer = sim.simxGetVisionSensorDepthBuffer(self.sim_client,
                                                                               cam_separate_handle,
                                                                               sim.simx_opmode_blocking)
        depth_img = np.asarray(depth_buffer)
        depth_img.shape = (resolution[1], resolution[0])
        zNear = 0.01
        zFar = 5
        depth_img = depth_img * (zFar - zNear) + zNear
        # depth_img = np.fliplr(depth_img)
        depth_img = np.flipud(depth_img)
        depth_img = depth_img * 1000
        depth_img = depth_img.astype('uint16')
        print('Got RGBD image from camera')

        return color_img, depth_img

    def set_acf_network(self, args):
        model = aff_cf_model.ACFNetwork(arch='resnet50', pretrained=True, num_classes=5, input_mode='RGBD',
                                        acf_head=args.acf_head)
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        if args.checkpoint_path != '':
            if not os.path.exists(args.checkpoint_path):
                os.system('mkdir -p ' + args.checkpoint_path)
            state_dict = torch.load(os.path.join(args.checkpoint_path, args.resume + '.pth'), map_location=device)
            model.load_state_dict(state_dict['model_state_dict'])
        model.to(device)
        model.eval()
        self.acf_model = model

    def network_forwardpass(self, image_rgb, image_depth_ori, out_dir=None, save_image_name='tmp.png'):
        image_depth = (image_depth_ori - image_depth_ori.min()) / (image_depth_ori.max() - image_depth_ori.min())
        image_rgb_tensor = TF.to_tensor(image_rgb).type(torch.float32)
        image_depth_tensor = TF.to_tensor(image_depth).type(torch.float32)
        img = torch.cat((image_rgb_tensor, image_depth_tensor), dim=0)
        img = img.unsqueeze(0)

        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        model = self.acf_model

        if out_dir is None:
            out_dir = self.args.out_dir
        out_imgdir = out_dir + '/result_images/'
        os.system('mkdir -p ' + out_imgdir)

        img = img.to(device)
        with torch.no_grad():
            detections, _ = model(img)
        proposals = acfutils.post_process_proposals(detections, image_depth_ori, K=config.BOX_POSTPROCESS_NUMBER)
        final_pafs_pair = acfutils.pafprocess(proposals, config.Unreal_camera_mat)
        acfutils.vis_images(proposals, [None], [image_rgb], self.classes, config.Unreal_camera_mat, training=False,
                            output_dir=out_imgdir, final_pafs_pair=final_pafs_pair, save_image_name=save_image_name)

        return proposals, final_pafs_pair

    def filter_proposals_by_gt(self, image_rgb, proposals, cam_pose, out_dir=None,
                               save_image_name='filtered_vrep.png', distance_tolerance=10, angle_tolerance=45):
        def get_part_groundtruth(sim_client, parent_pose, part_name, cam_pose):
            _, local_parent_handle = sim.simxGetObjectHandle(sim_client, name, sim.simx_opmode_blocking)

            local_body_name = ('Dummy_' + name).replace('model', part_name)
            _, local_body_handle = sim.simxGetObjectHandle(sim_client, local_body_name, sim.simx_opmode_blocking)
            _, local_body_pos = sim.simxGetObjectPosition(sim_client, local_body_handle, local_parent_handle,
                                                          sim.simx_opmode_blocking)
            _, local_body_rot = sim.simxGetObjectOrientation(sim_client, local_body_handle, local_parent_handle,
                                                             sim.simx_opmode_blocking)
            rot_matrix = euler2rotm(local_body_rot)
            local_mat = np.eye(4)
            local_mat[:3, 3] = np.array(local_body_pos)
            local_mat[:3, :3] = rot_matrix
            world_pose = np.dot(parent_pose, local_mat)

            return np.dot(np.linalg.inv(cam_pose), world_pose)

        groundtruth_part = {'body': [], 'handle': [], 'stir': [], 'head': []}
        id_to_model = {i['handle']: i['name'] for i in self.object_list_}
        for ob in self.object_list_:
            name = ob['name']
            handle = ob['handle']
            _, pos = sim.simxGetObjectPosition(self.sim_client, handle, -1, sim.simx_opmode_blocking)
            _, rot = sim.simxGetObjectOrientation(self.sim_client, handle, -1, sim.simx_opmode_blocking)
            pose_mat = np.eye(4)
            pose_mat[:3, 3] = np.array(pos)
            pose_mat[:3, :3] = euler2rotm(rot)
            if 'Bottle' in name or 'Bowl' in name:
                body_camera_pose = get_part_groundtruth(self.sim_client, pose_mat, 'body', cam_pose)
                groundtruth_part['body'].append(
                    list(body_camera_pose[:3, 3] * 1e2) + list(body_camera_pose[:3, 2]) + [handle])
            elif 'Mug' in name:
                body_camera_pose = get_part_groundtruth(self.sim_client, pose_mat, 'body', cam_pose)
                groundtruth_part['body'].append(
                    list(body_camera_pose[:3, 3] * 1e2) + list(body_camera_pose[:3, 2]) + [handle])
                handle_camera_pose = get_part_groundtruth(self.sim_client, pose_mat, 'handle', cam_pose)
                groundtruth_part['handle'].append(
                    list(handle_camera_pose[:3, 3] * 1e2) + list(handle_camera_pose[:3, 2]) + [handle])
            elif 'Spoon' in name:
                stir_camera_pose = get_part_groundtruth(self.sim_client, pose_mat, 'stir', cam_pose)
                groundtruth_part['stir'].append(
                    list(stir_camera_pose[:3, 3] * 1e2) + list(stir_camera_pose[:3, 2]) + [handle])
                head_camera_pose = get_part_groundtruth(self.sim_client, pose_mat, 'head', cam_pose)
                groundtruth_part['head'].append(
                    list(head_camera_pose[:3, 3] * 1e2) + list(head_camera_pose[:3, 2]) + [handle])

        for p in proposals:
            total_part_indexs = torch.arange(p['keypoints_3d'].shape[0])
            labels = p['labels']
            keypoints_3d = p['keypoints_3d']
            axis = p['axis']
            assign_index = []
            groundtruth_ids = []
            distance_errors = []
            angle_errors = []
            for part_name in groundtruth_part:
                part_id = self.classes[part_name]
                selected_label = labels == part_id
                selected_index = total_part_indexs[selected_label]
                proposal_keypoint = keypoints_3d[selected_label]
                proposal_axis = axis[selected_label]
                if len(groundtruth_part[part_name]) <= selected_index.shape[0]:
                    unassigned_proposal = torch.ones(proposal_keypoint.shape[0], out=selected_label)
                    unassigned_proposal_index = torch.arange(proposal_keypoint.shape[0])
                    for gt_pose_with_id in groundtruth_part[part_name]:
                        gt_pose = torch.tensor(gt_pose_with_id[:-1], dtype=proposal_keypoint.dtype,
                                               device=proposal_keypoint.device)
                        if torch.nonzero(unassigned_proposal).shape[0] == 0:
                            continue
                        distance = torch.norm(proposal_keypoint[unassigned_proposal] - gt_pose[:3], dim=-1)
                        index = torch.argmin(distance)
                        v = 180.0 / math.pi * torch.acos(
                            torch.clamp(proposal_axis[unassigned_proposal].mm(gt_pose[3:, None]), -1 + 1e-9, 1 - 1e-9))

                        print(
                            'obj:{}, part:{}, distance:{}, angle:{}'.format(id_to_model[gt_pose_with_id[-1]], part_name,
                                                                            distance[index], v[index][0]))
                        if distance[index] <= distance_tolerance and v[index] <= angle_tolerance:
                            distance_errors.append(distance[index].item())
                            angle_errors.append(v[index][0].item())
                            u_i = unassigned_proposal_index[unassigned_proposal]
                            unassigned_proposal[u_i[index]] = 0
                            assign_index.append(selected_index[u_i[index]].item())
                            groundtruth_ids.append(gt_pose_with_id[-1])
                else:
                    unassigned_proposal = torch.ones(len(groundtruth_part[part_name]), out=selected_label)
                    unassigned_proposal_index = torch.arange(len(groundtruth_part[part_name]))
                    for i, k in enumerate(proposal_keypoint):
                        gt_pose = torch.tensor(np.array(groundtruth_part[part_name]), dtype=proposal_keypoint.dtype,
                                               device=proposal_keypoint.device)
                        if torch.nonzero(unassigned_proposal).shape[0] == 0:
                            continue
                        distance = torch.norm(k - gt_pose[unassigned_proposal, :3], dim=-1)
                        index = torch.argmin(distance)
                        v = 180.0 / math.pi * torch.acos(
                            torch.clamp(gt_pose[unassigned_proposal, 3:6].mm(proposal_axis[i][:, None]), -1 + 1e-9,
                                        1 - 1e-9))

                        print(
                            'obj:{}, part:{}, distance:{}, angle:{}'.format(
                                int(gt_pose[unassigned_proposal, 6][index].item()), part_name,
                                distance[index], v[index][0]))
                        if distance[index] <= distance_tolerance and v[index] <= angle_tolerance:
                            distance_errors.append(distance[index].item())
                            angle_errors.append(v[index][0].item())
                            u_i = unassigned_proposal_index[unassigned_proposal]
                            groundtruth_ids.append(int(gt_pose[unassigned_proposal, 6][index].item()))
                            unassigned_proposal[u_i[index]] = 0
                            assign_index.append(selected_index[i].item())
            for i in p:
                if torch.is_tensor(p[i]):
                    p[i] = p[i][assign_index]
            p['groundtruth_ids'] = groundtruth_ids
            p['distance_errors'] = distance_errors
            p['angle_errors'] = angle_errors

        if out_dir is None:
            out_dir = self.args.out_dir
        out_imgdir = out_dir + '/result_images/'
        os.system('mkdir -p ' + out_imgdir)

        final_pafs_pair = acfutils.pafprocess(proposals, config.Unreal_camera_mat)
        acfutils.vis_images(proposals, [None], [image_rgb], self.classes, config.Unreal_camera_mat, training=False,
                            output_dir=out_imgdir, final_pafs_pair=final_pafs_pair, save_image_name=save_image_name)

        return proposals, final_pafs_pair

    def proposal_to_grasp(self, proposals, final_pafs_pair, cam_pose):
        id_to_model = {i['handle']: i['name'] for i in self.object_list_}
        grasp_pose = {i['handle']: [] for i in self.object_list_}
        error_dict = {i['handle']: [] for i in self.object_list_}
        for (p, f) in zip(proposals, final_pafs_pair):
            gt_id = p['groundtruth_ids']
            keypoints = p['keypoints_3d'].numpy()
            keypoints = cam_pose[:3, :3].dot(keypoints.transpose()) + cam_pose[:3, 3, None] * 1e2
            keypoints = keypoints.transpose()
            axis = p['axis'].numpy()
            axis = cam_pose[:3, :3].dot(axis.transpose()).transpose()
            labels = p['labels'].numpy()
            distance_errors = p['distance_errors']
            angle_errors = p['angle_errors']
            for i in range(len(gt_id)):
                k = list(keypoints[i])
                a = list(axis[i])
                label = [labels[i]]
                grasp_pose[gt_id[i]].append(k + a + label + [i])
                error_dict[gt_id[i]].append([distance_errors[i]] + [angle_errors[i]] + label)
            for id in grasp_pose:
                if len(grasp_pose[id]) != 0:
                    pose = np.array(grasp_pose[id])
                else:
                    pose = []
                if 'Mug' in id_to_model[id] or 'Spoon' in id_to_model[id]:
                    if len(grasp_pose[id]) >= 2:
                        find = False
                        p1 = set(pose[:, -1])
                        for pair in f:
                            p2 = set(pair[:2])
                            if p1 == p2:
                                find = True
                                break
                        if not find:
                            pose = []
                    else:
                        pose = []
                if pose == []:
                    grasp_pose[id] = pose
                else:
                    grasp_pose[id] = pose[:, :-1]
        return grasp_pose, error_dict

    def perception_results(self, out_dir=None, image_save_name=''):
        # by default only process first camera, if use_second_camera is True, need also pass second_camera_id and pose
        color_img, depth_img = self.get_camera_data(self.cam_handle)
        if out_dir is not None:
            cv2.imwrite(os.path.join(out_dir, 'camer1_color.png'), cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(out_dir, 'camer1_depth.png'), depth_img)
        image_raw_name = image_save_name + '_raw.png'
        image_filter_name = image_save_name + '_filter.png'
        proposals, _ = self.network_forwardpass(color_img, depth_img, out_dir=out_dir, save_image_name=image_raw_name)
        proposals, paf_pairs = self.filter_proposals_by_gt(color_img, proposals, self.cam_pose, out_dir=out_dir,
                                                           save_image_name=image_filter_name)
        estimate_pose, error_dict = self.proposal_to_grasp(proposals, paf_pairs, self.cam_pose)
        if self.args.use_second_camera:
            color_img, depth_img = self.get_camera_data(self.cam_handle2)
            if out_dir is not None:
                cv2.imwrite(os.path.join(out_dir, 'camer2_color.png'), cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join(out_dir, 'camer2_depth.png'), depth_img)
            proposals, _ = self.network_forwardpass(color_img, depth_img, out_dir=out_dir,
                                                    save_image_name=image_raw_name[:-4] + '_2nd.png')
            proposals, paf_pairs = self.filter_proposals_by_gt(color_img, proposals, self.cam_pose2, out_dir=out_dir,
                                                               save_image_name=image_filter_name[:-4] + '_2nd.png')
            second_estimate_pose, second_error_dict = self.proposal_to_grasp(proposals, paf_pairs, self.cam_pose2)
            for id in estimate_pose:
                if estimate_pose[id] == [] and second_estimate_pose[id] != []:
                    estimate_pose[id] = second_estimate_pose[id]
                    error_dict[id] = second_error_dict[id]
        return estimate_pose, error_dict

    # # core manipulation routine functions
    def grasp_pipeline(self, estimate_pose=None, error_dict=None, obj=None, out_dir=None):
        self.update_collision_set()
        results_save_name = 'grasp_pipeline.txt'
        if out_dir is not None:
            results_save_name = os.path.join(out_dir, results_save_name)
        succ = False
        if obj is None:
            obj = self.object_list_[-1]
        if estimate_pose is not None:
            grasp_txt = open(results_save_name, 'a+')
            grasp_pose_list = self.calculate_grasp_pose_with_perception(obj, estimate_pose)
            if not grasp_pose_list:
                print('cannot find grasp pose')
                # self.remove_object(obj)
                self.object_list_.remove(obj)
                grasp_txt.write('None, None, False, {}_undetected\n'.format(obj['name']))
            else:
                succ, _ = self.grasp(obj, grasp_pose_list, drop_obj=True, reset_arm=True, lift=True)
                error = np.array(error_dict[obj['handle']])
                if 'Mug' in obj['name']:
                    h_index = np.where(error[:, -1] == 2)[0]
                elif 'Bottle' in obj['name']:
                    h_index = np.where(error[:, -1] == 1)[0]
                elif 'Spoon' in obj['name']:
                    h_index = np.where(error[:, -1] == 3)[0]
                grasp_txt.write(
                    '{:.3f}, {:.3f}, {}, {}\n'.format(error[h_index[0], 0], error[h_index[0], 1], succ, obj['name']))
            grasp_txt.close()
        else:
            grasp_pose_list = self.calculate_grasp_pose(obj, self.get_obj_pose(obj['handle']))
            succ, _ = self.grasp(obj, grasp_pose_list, drop_obj=True, reset_arm=True, lift=True)
        return succ

    def pour_pipeline(self, container, pourer, estimate_pose=None, color=None):
        """
        1. generate particles to one container
        2. calculate grasp pose of this container, and goal position above the other (should reserve a certain distance for pour)
        3. grasp
        4. move to the goal position
        5. rotate to pour
        :return:
        """
        self.update_collision_set()
        is_bottle = 'Bottle' in pourer['name']
        self.add_particles(pourer, color=color)
        if estimate_pose is None:
            obj_pose = self.get_obj_pose(pourer['handle'])
            grasp_pose_list = self.calculate_grasp_pose(pourer, obj_pose)
        else:
            grasp_pose_list = self.calculate_grasp_pose_with_perception(pourer, estimate_pose)
        if grasp_pose_list == []:
            print('not detected or associated object parts')
            return 4
        weight_pourer_before = self.read_weight(pourer)
        succ, grasp_pose = self.grasp(pourer, grasp_pose_list, remove_obj=False, reset_arm=False)
        if not succ:
            print('cannot find grasp pose')
            return 1
        path2grasp = self.path_record  # save path before grasping, for return the pourer to original place
        self.path_record = []
        if estimate_pose is None:
            pour_pose_list = self.calculate_pour_pose(container['handle'], container['name'], pourer['handle'],
                                                      pourer['name'], grasp_pose, is_bottle)
        else:
            pour_pose_list = self.calculate_pour_pose_with_perception(container['handle'], pourer['handle'],
                                                                      estimate_pose, grasp_pose, is_bottle)
        if pour_pose_list == []:
            print('container body or pourer body not detected')
            return 4
        interp_pose = deepcopy(grasp_pose)
        interp_pose[2, 3] = pour_pose_list[0][2, 3] - 0.1  # lift first
        self.cartesian_move(interp_pose, msg='cannot find way to lift container', min_configs_ik_path=2)
        foundPath = False
        for pour_pose in pour_pose_list:  # try cartesian move first, motion plan may spill water outside
            if self.cartesian_move(pour_pose, min_configs_ik_path=2):
                foundPath = True
                break
        if not foundPath:
            for pour_pose in pour_pose_list:
                if self.cartesian_move(pour_pose, min_configs_ik_path=2, try_motion_plan=True):
                    foundPath = True
                    break
        if not foundPath:
            print('cannot find way to reach above the other container')
            return 2

        # if not is_bottle: # for large containers, this height might still not be enough, try to move higher if possible
        #     pour_pose[2, 3] += 0.05
        #     self.cartesian_move(pour_pose, msg='cannot find way to lift container', min_configs_ik_path=2)
        clockwise = grasp_pose[2, 1] > 0
        weight_container_before = self.read_weight(container)
        self.pour(clockwise)
        # n_particles_poured = self.count_particles(container_2['name']) # TODO: uncomment this line after implementation
        weight_container_after = self.read_weight(container)
        self.reset_arm()  # place the object to original place, then return the arm to default configuration
        self.open_gripper(is_bottle)
        self.attachdetach2gripper(pourer, 0)
        self.path_record = path2grasp
        self.reset_arm()
        weight_pourer_after = self.read_weight(pourer)
        print('pour pipeline finished!')
        pour_weight_ratio = (weight_container_after - weight_container_before) / (
                weight_pourer_before - weight_pourer_after)
        print('pour ratio: ', pour_weight_ratio)
        if pour_weight_ratio > self.pour_ratio_threshold:
            return 0
        else:
            return 3

    def stir_pipeline(self, container, spoon, estimate_pose=None, add_particle=True):
        """
        1. generate a random container and a random spoon
        2. calculate grasp pose of the spoon, lift it,
        3. move it above the container body keypoint, get some inclination (should reserve a certain distance for stir), generate two kinds of particles in the container
        4. go downwards to insert the spoon into the container, reach the head keypoint to the body keypoint
        5. random or simple trajectory to stir
        6. go upwards to pull the spoon
        :return:
        """
        self.update_collision_set()
        lift_distance = 0.2
        if estimate_pose is None:
            obj_pose = self.get_obj_pose(spoon['handle'])
            grasp_pose_list = self.calculate_grasp_pose(spoon, obj_pose)
        else:
            grasp_pose_list = self.calculate_grasp_pose_with_perception(spoon, estimate_pose)
        succ, grasp_pose = self.grasp(spoon, grasp_pose_list, remove_obj=False, reset_arm=False)
        if not succ:
            print('cannot find grasp pose of spoon')
            return 1
        path2grasp = self.path_record  # save path before grasping, for return the pourer to original place
        self.path_record = []
        if estimate_pose is None:
            hang_pose_list, head2grasp = self.calculate_stir_pose(spoon['name'], spoon['handle'], grasp_pose,
                                                                  container['name'], container['handle'])
        else:
            hang_pose_list, head2grasp = self.calculate_stir_pose_with_perception(estimate_pose, spoon['handle'],
                                                                                  grasp_pose, container['handle'])
        interp_pose = deepcopy(grasp_pose)
        interp_pose[2, 3] = hang_pose_list[0][2, 3]  # try to lift first
        self.cartesian_move(interp_pose, msg='cannot find way to lift spoon first', try_motion_plan=True,
                            min_configs_ik_path=2)
        interp_pose = deepcopy(grasp_pose)
        interp_pose[:3, 3] = hang_pose_list[0][:3, 3]  # try to reach the position with the same orientation first
        self.cartesian_move(interp_pose, msg='cannot find way to reach the position first', try_motion_plan=True,
                            min_configs_ik_path=2)
        foundPath = False
        for hang_pose in hang_pose_list:
            if self.cartesian_move(hang_pose, try_motion_plan=True, min_configs_ik_path=2):
                time.sleep(3)
                foundPath = True
                break
        if not foundPath:
            print('cannot find way to reach above the other container')
            # return 2
        if add_particle:
            self.add_particles(container)
            self.add_particles(container, color=[1, 1, 0], offset=[0, 0, 0.01])
        # self.set_object_dynamic(spoon['handle'], 1)
        stir_pose = deepcopy(hang_pose)
        stir_pose[2, 3] -= (lift_distance - self.stir_height_offset)  # do not let spoon touch the bottom of container
        if not self.cartesian_move(stir_pose, msg='cannot find way to move spoon downwards', try_motion_plan=True,
                                   min_configs_ik_path=2, try_ik=foundPath):
            return 3
        time.sleep(3)
        res = self.check_particle_mix(container, spoon)
        print('stir position check: {}'.format(res))
        self.stir(stir_pose, head2grasp)
        self.reset_arm()  # place the object to original place, then return the arm to default configuration
        self.open_gripper()
        self.attachdetach2gripper(spoon, 0)
        self.path_record = path2grasp
        self.reset_arm()
        print('stir pipeline finished!')
        if res:
            return 0
        else:
            return 4

    def drinkserve_pipeline(self, container, spoon, pourers, estimate_pose=None, error_dict=None, error_save_dir=None):
        txt = open(os.path.join(error_save_dir, 'drinking_pipeline.txt'), 'a+')
        particle_colors = [[0, 1, 1], [1, 1, 0]]
        error = []
        id_to_model = {i['handle']: i['name'] for i in self.object_list_}
        for id in estimate_pose:
            if estimate_pose[id] == []:
                txt.write('None, None, False, {}_undetected \n'.format(id_to_model[id]))
                print('{} undetected'.format(id_to_model[id]))
                return False
        for i, (pourer, pcolor) in enumerate(zip(pourers, particle_colors)):
            pour_return = self.pour_pipeline(container, pourer, estimate_pose, pcolor)
            if error_dict is not None:
                container_error = error_dict[container['handle']]
                pourer_error = error_dict[pourer['handle']]
                if pour_return != 0:
                    self.clear_workspace()
                    pour_error = np.array(container_error + pourer_error)
                    txt.write('{:.3f}, {:.3f}, False, times:{}_pour_{}\n'.format(pour_error[:, 0].mean(),
                                                                                 pour_error[:, 1].mean(), i,
                                                                                 pour_return))
                    txt.close()
                    print('pour pipeline failed using {}, {}'.format(pourer['name'], container['name']))
                    return False
                error += pourer_error

        stir_return = self.stir_pipeline(container, spoon, add_particle=False)
        if error_dict is not None:
            container_error = error_dict[container['handle']]
            spoon_error = error_dict[spoon['handle']]
            if stir_return == 0:
                error += container_error + spoon_error
                error = np.array(error)
                txt.write('{:.3f}, {:.3f}, True, {}\n'.format(error[:, 0].mean(), error[:, 1].mean(), stir_return))
            else:
                stir_error = np.array(container_error + spoon_error)
                txt.write('{:.3f}, {:.3f}, False, stir_{}\n'.format(stir_error[:, 0].mean(), stir_error[:, 1].mean(),
                                                                    stir_return))
        txt.close()
        self.clear_workspace()
        return True

    # # ACF calculation
    def grasp_pose_acf(self, obj, part1_acf, part2_acf):
        if 'Mug' in obj['name']:  # grasp by handle, 2 grasp pose candidates from two sides of handle
            grasp_z_axis = np.cross(part2_acf[1], part1_acf[1])
            grasp_z_axis = grasp_z_axis / np.linalg.norm(grasp_z_axis)
            grasp_y_axis = np.cross(grasp_z_axis, part1_acf[1])
            grasp_y_axis = grasp_y_axis / np.linalg.norm(grasp_y_axis)
            grasp_x_axis = np.cross(grasp_y_axis, grasp_z_axis)
            grasp_poses = [np.vstack((np.vstack((grasp_x_axis, grasp_y_axis, grasp_z_axis, np.array(part1_acf[0]))).T,
                                      np.array([0, 0, 0, 1])))]
            rot_angle = 10 / 180 * math.pi
            for n in [-2, -1, 1, 2]:
                angle = rot_angle * n
                grasp_poses.append(np.dot(grasp_poses[0], np.array([[np.cos(angle), -np.sin(angle), 0, 0],
                                                                    [np.sin(angle), np.cos(angle), 0, 0],
                                                                    [0, 0, 1, 0], [0, 0, 0, 1]])))
                grasp_poses.append(np.dot(grasp_poses[0], np.array([[np.cos(angle), 0, np.sin(angle), 0], [0, 1, 0, 0],
                                                                    [-np.sin(angle), 0, np.cos(angle), 0],
                                                                    [0, 0, 0, 1]])))
            n_sample = len(grasp_poses)
            for i in range(n_sample):
                grasp_poses.append(
                    np.dot(grasp_poses[i], np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])))
        elif 'Spoon' in obj['name']:  # grasp by stir, 4 grasp pose candidates, 90 degree interval, toward stir
            grasp_z_axis, grasp_y_axis = -part2_acf[1], part1_acf[1]
            grasp_x_axis = np.cross(grasp_y_axis, grasp_z_axis)
            grasp_poses = [np.vstack((np.vstack((grasp_x_axis, grasp_y_axis, grasp_z_axis, np.array(part1_acf[0]))).T,
                                      np.array([0, 0, 0, 1])))]
            for i in range(8 - 1):  # change to 8 poses around
                grasp_poses.append(
                    np.dot(grasp_poses[-1], np.array([[math.sqrt(2) / 2, 0, math.sqrt(2) / 2, 0], [0, 1, 0, 0],
                                                      [-math.sqrt(2) / 2, 0, math.sqrt(2) / 2, 0], [0, 0, 0, 1]])))
            for i in range(8):  # also try poses near to handle keypoint with same orientation
                pose = deepcopy(grasp_poses[i])
                pose[:3, 3] += 0.03 * pose[:3, 1]  # translate along y axis of grasp pose
                grasp_poses.append(pose)
                pose[:3, 3] -= 2 * 0.03 * pose[:3, 1]
                grasp_poses.append(pose)
        elif 'Bottle' in obj['name']:  # grasp by body
            grasp_y_axis = part1_acf[1]
            grasp_z_axis = np.cross(grasp_y_axis, np.random.rand(3, ))
            grasp_z_axis = grasp_z_axis / np.linalg.norm(grasp_z_axis)
            grasp_x_axis = np.cross(grasp_y_axis, grasp_z_axis)
            grasp_poses = [np.vstack((np.vstack((grasp_x_axis, grasp_y_axis, grasp_z_axis, np.array(part1_acf[0]))).T,
                                      np.array([0, 0, 0, 1])))]
            for i in range(8 - 1):  # change to 8 poses around
                grasp_poses.append(
                    np.dot(grasp_poses[-1], np.array([[math.sqrt(2) / 2, 0, math.sqrt(2) / 2, 0], [0, 1, 0, 0],
                                                      [-math.sqrt(2) / 2, 0, math.sqrt(2) / 2, 0], [0, 0, 0, 1]])))
        return grasp_poses

    def calculate_grasp_pose(self, obj, obj_pose):
        if 'Mug' in obj['name']:
            grasp_part, second_part = self.object_lib[obj['name']]['handle'], self.object_lib[obj['name']]['body']
        elif 'Spoon' in obj['name']:
            grasp_part, second_part = self.object_lib[obj['name']]['stir'], self.object_lib[obj['name']]['head']
        elif 'Bottle' in obj['name']:  # grasp by body
            grasp_part, second_part = self.object_lib[obj['name']]['body'], None
        else:
            print('grasp pose not implemented for bowls')
            return []
        grasp_poses = self.grasp_pose_acf(obj, grasp_part, second_part)
        grasp_poses = [np.dot(obj_pose, grasp_pose) for grasp_pose in grasp_poses]
        return grasp_poses

    def calculate_grasp_pose_with_perception(self, obj, estimate_pose):
        pose = estimate_pose[obj['handle']]
        if pose == []:
            return []
        if 'Mug' in obj['name']:  # grasp by handle, 2 grasp pose candidates from two sides of handle
            grasp_part = self.estimate_pose2acf(estimate_pose, obj['handle'], 'handle')
            second_part = self.estimate_pose2acf(estimate_pose, obj['handle'], 'body')
        elif 'Spoon' in obj['name']:  # grasp by stir, 4 grasp pose candidates, 90 degree interval, toward stir
            grasp_part = self.estimate_pose2acf(estimate_pose, obj['handle'], 'stir')
            second_part = self.estimate_pose2acf(estimate_pose, obj['handle'], 'head')
        elif 'Bottle' in obj['name']:  # grasp by body
            grasp_part, second_part = self.estimate_pose2acf(estimate_pose, obj['handle'], 'body'), None
        else:
            print('grasp pose not implemented for bowls')
            return []
        grasp_poses = self.grasp_pose_acf(obj, grasp_part, second_part)
        return grasp_poses

    def calculate_pour_pose(self, container_handle, container_name, pourer_handle, pourer_name, grasp_pose, is_bottle):
        part1_acf = self.gt_obj2acf(container_name, container_handle, 'body')
        part2_acf = self.gt_obj2acf(pourer_name, pourer_handle, 'body')
        return pour_pose_acf(is_bottle, part1_acf, part2_acf, grasp_pose)

    def calculate_pour_pose_with_perception(self, container_handle, pourer_handle, estimate_pose, grasp_pose,
                                            is_bottle):
        try:
            part1_acf = self.estimate_pose2acf(estimate_pose, container_handle, 'body')
            part2_acf = self.estimate_pose2acf(estimate_pose, pourer_handle, 'body')
            return pour_pose_acf(is_bottle, part1_acf, part2_acf, grasp_pose)
        except:
            return []

    def calculate_stir_pose(self, spoon_name, spoon_handle, grasp_pose, container_name, container_handle):
        # make spoon straight downwards right above the container
        part1_acf = self.gt_obj2acf(container_name, container_handle, 'body')
        part2_acf = self.gt_obj2acf(spoon_name, spoon_handle, 'head')
        part3_acf = self.gt_obj2acf(spoon_name, spoon_handle, 'stir')
        return stir_pose_acf(part1_acf, part2_acf, part3_acf, grasp_pose)

    def calculate_stir_pose_with_perception(self, estimate_pose, spoon_handle, grasp_pose, container_handle):
        try:
            part1_acf = self.estimate_pose2acf(estimate_pose, container_handle, 'body')
            part2_acf = self.estimate_pose2acf(estimate_pose, spoon_handle, 'head')
            part3_acf = self.estimate_pose2acf(estimate_pose, spoon_handle, 'stir')
            return stir_pose_acf(part1_acf, part2_acf, part3_acf, grasp_pose)
        except:
            return [], []

    # # low-level manipulation interface
    def grasp(self, obj, grasp_pose_list, remove_obj=False, reset_arm=False, lift=False, drop_obj=False):
        print('try to grasp', obj['name'])
        pregrasp_step = 0.05
        lift_step = 0.05
        succ = False
        if 'Bottle' in obj['name']:
            open_mode = 1
        else:
            open_mode = 0
        self.open_gripper(open_mode=open_mode)
        grasp_pose = None
        for grasp_pose in grasp_pose_list:
            self.path_record = []
            if grasp_pose[2, 3] < self.tabletop_z:  # do not try grasp pose candidates lower than tabletop
                print('grasp pose candidate lower than tabletop')
                continue
            self.show_pose(grasp_pose)
            offset_mat = np.zeros((4, 4))
            offset_mat[:3, -1] = pregrasp_step * grasp_pose[:3, 2]
            pre_grasp_pose = grasp_pose - offset_mat
            self.show_pose(pre_grasp_pose)
            if not self.cartesian_move(pre_grasp_pose, msg='cannot find way to pre grasp pose', try_motion_plan=True,
                                       try_ik=False):
                continue
            if not self.cartesian_move(grasp_pose, msg='cannot find way to grasp pose'):
                break
            time.sleep(2)
            self.close_gripper()
            time.sleep(1)
            succ = self.check_object_grasped(obj['name'], obj['handle'])
            if succ:
                self.attachdetach2gripper(obj)
                if 'Spoon' in obj['name']:
                    self.set_object_respondable(obj['handle'], 0)
                    # self.set_object_dynamic(obj['handle'], 1)
                else:
                    self.set_object_dynamic(obj['handle'], 1)
                print('grasp success')
                if not drop_obj and lift:
                    lift_pose = deepcopy(grasp_pose)
                    lift_pose[2, 3] += lift_step
                    self.cartesian_move(lift_pose, msg='cannot find way to lift')
                else:
                    time.sleep(2)
                succ = self.check_object_grasped(obj['name'], obj['handle'])  # double check grasping
            if not succ:
                print('failed grasp checking')
            break
        # return arm to original state, delete tested object
        if drop_obj:
            if succ:
                self.drop_object(obj)
                self.force_reset_arm()
            else:
                self.attachdetach2gripper(obj, attach=0)
                self.open_gripper()
                self.reset_arm()
        else:
            self.open_gripper()
            if remove_obj:
                self.remove_object(obj)
            if reset_arm:
                self.reset_arm()
        return succ, grasp_pose

    def pour(self, clockwise):  # clockwise refers to last joint rotation direction from the view of its parent link
        current_joint_angles = self.get_joint_angles()
        desired_joint_angles_1 = deepcopy(current_joint_angles)
        desired_joint_angles_2 = deepcopy(current_joint_angles)
        if clockwise:  # tested, joint7 is for rotating
            desired_joint_angles_1[-1] -= math.pi * 4 / 9
            desired_joint_angles_2[-1] -= math.pi * 5 / 9
        else:
            desired_joint_angles_1[-1] += math.pi * 4 / 9
            desired_joint_angles_2[-1] += math.pi * 5 / 9
        path = np.linspace(current_joint_angles, desired_joint_angles_1, num=10).flatten().tolist() + \
               np.linspace(desired_joint_angles_1, desired_joint_angles_2, num=50).flatten().tolist()
        self.execute(path)
        time.sleep(10)
        # path = np.linspace(desired_joint_angles, current_joint_angles, num=10).flatten().tolist()
        # self.execute(path)

    def stir(self, init_stir_pose, head2grasp, n_stir=1):
        # go forward and backward along head axis in horizontal plane
        stir_distance = 0.01
        stir_pose_forward = deepcopy(init_stir_pose)
        init_head_pose = np.dot(init_stir_pose, np.linalg.inv(head2grasp))
        stir_axis = init_head_pose[:3, 2]
        xy_vector_norm = np.linalg.norm(stir_axis[:2])
        stir_pose_forward[0, 3] += stir_axis[0] / xy_vector_norm * stir_distance / 2
        stir_pose_forward[1, 3] += stir_axis[1] / xy_vector_norm * stir_distance / 2
        stir_pose_backward = 2 * init_stir_pose - stir_pose_forward
        foundPath, path = self.generate_ik_path(stir_pose_forward)
        if not foundPath:
            print('cannot perform stir actions')
            return
        self.execute(path)
        for i in range(n_stir):
            foundPath, path = self.generate_ik_path(stir_pose_backward)
            if foundPath:
                self.execute(path)
            foundPath, path = self.generate_ik_path(stir_pose_forward)
            if foundPath:
                self.execute(path)

    def cartesian_move(self, pose, try_motion_plan=False, try_ik=True, msg='', min_configs_ik_path=10):
        foundPath = False
        if try_ik:
            foundPath, path = self.generate_ik_path(pose, min_configs_ik_path)
        if foundPath:
            self.execute(path)
        elif try_motion_plan:
            foundPath, path = self.plan(pose)
            if foundPath:
                if not path:
                    foundPath = False
                else:
                    self.execute(path)
        if not foundPath and msg != '':
            print(msg)
        return foundPath

    def plan(self, target_pose, pathfilename='latest_path', load_path=0, save_path=0):
        # not used
        approach_vector = [0, 0, 0.05]
        max_configs_desired_pose = 10  # we will try to find 10 different states corresponding to the goal pose
        max_trials_config_search = 300  # a parameter needed for finding appropriate goal states
        search_count = 30  # how many times OMPL will run for a given task
        min_configs_path_planning = 400  # interpolation states for the OMPL path
        min_configs_ik_path = 100  # interpolation states for the linear approach path
        collision_checking = 1  # whether collision checking is on or off

        # Do the path planning here (between a start state and a goal pose, including a linear approach phase):
        inInts = [self.robot_handle, collision_checking, min_configs_ik_path, min_configs_path_planning,
                  max_configs_desired_pose, max_trials_config_search, search_count, load_path, save_path]

        inFloats = target_pose.flatten().tolist() + approach_vector
        self.call_remote('findPath_goalIsPose', int_param=inInts, float_param=inFloats, str_param=[pathfilename],
                         mode=sim.simx_opmode_oneshot)

        while 1:
            res, retInt = sim.simxGetIntegerSignal(self.sim_client, 'Planner_finished', sim.simx_opmode_oneshot_wait)
            if res == 1 or retInt == 0:
                continue
            elif retInt == -1:
                return False, []
            elif retInt == 1:
                _, _, retFloats, _, _ = self.call_remote('get_path')
                return True, retFloats

    def execute(self, path, load_file=False, load_filename=''):
        line_handle = self.visualize_path(path)
        self.path_record += path
        print('follow the path')
        # Make the robot follow the path:
        if load_file:
            self.call_remote('runThroughPath', int_param=[self.robot_handle, 1], str_param=[load_filename])
        else:
            self.call_remote('runThroughPath', int_param=[self.robot_handle, 0], float_param=path)

        # Wait until the end of the movement:
        running_path = True
        while running_path:
            _, retInts, _, _, _ = self.call_remote('isRunningThroughPath', int_param=[self.robot_handle])
            if retInts[0] == 0:
                running_path = False
        self.clear_visualization_path(line_handle)

    def generate_ik_path(self, target_pose, min_configs_ik_path=10):
        collision_checking = 0  # whether collision checking is on or off
        # Do the path planning here (between a start state and a goal pose, including a linear approach phase):
        _, retInts, path, _, _ = self.call_remote('generateIkPath',
                                                  int_param=[self.robot_handle, collision_checking,
                                                             min_configs_ik_path],
                                                  float_param=target_pose.flatten().tolist()[:12])
        return retInts[0], path

    # # utilities
    def call_remote(self, function_name, int_param=None, float_param=None, str_param=None,
                    mode=sim.simx_opmode_blocking):
        if str_param is None:
            str_param = []
        if float_param is None:
            float_param = []
        if int_param is None:
            int_param = []
        res, retInts, retFloats, retStrings, retBuffer = sim.simxCallScriptFunction(self.sim_client,
                                                                                    'remoteApiCommandServer',
                                                                                    sim.sim_scripttype_childscript,
                                                                                    function_name,
                                                                                    int_param, float_param, str_param,
                                                                                    self.empty_buff,
                                                                                    mode)
        return res, retInts, retFloats, retStrings, retBuffer

    def add_particles(self, obj, color=None, grid_size=None, offset=None, size=0.01, times=1):
        # generate square grid with edge length = number, repeat times = times
        obj_name = obj['name']
        obj_body_acf = self.gt_obj2acf(obj_name, obj['handle'], 'body')
        if grid_size is None:
            grid_size = self.particle_grid_size
        if 'Bottle' in obj_name:
            times = 2
        elif 'Mug' in obj_name:
            times = 3
        elif 'Bowl' in obj_name:
            times = 4
        if color is None:
            color = [0, 1, 1]
        if offset is None:
            offset = [0, 0, 0]
        for i in range(times):
            _, phandle, _, _, _ = self.call_remote('addParticles', int_param=color + grid_size,
                                                   float_param=[size] + offset + list(obj_body_acf[0]))
            self.particle_objects.append(phandle[0])
            time.sleep(2)
        return grid_size[0] * grid_size[1] * times

    def clear_workspace(self):
        self.remove_particles()
        while len(self.object_list_) > 0:
            self.remove_object(self.object_list_[-1])
        self.reset_arm()
        self.open_gripper()
        if not self.check_arm_returned():
            self.force_reset_arm()

    def read_arm_joints(self):
        _, _, retFloats, _, _ = self.call_remote('getJointPositions')
        return retFloats

    def check_arm_returned(self):
        _, _, retFloats, _, _ = self.call_remote('getJointPositions')
        for joint, init_joint in zip(retFloats, self.init_arm_joints):
            if abs(joint - init_joint) > 0.1:
                return False
        return True

    def force_reset_arm(self):
        path = np.linspace(self.read_arm_joints(), self.init_arm_joints, num=100).flatten().tolist()
        self.execute(path)
        self.path_record = []
        time.sleep(2)

    def remove_particles(self):
        for phandle in self.particle_objects:
            self.call_remote('removeParticles', [phandle])
        self.particle_objects = []

    def generate_random_objects(self, n_objs=None):
        if n_objs is None:
            n_objs = {'spoon': [1, 3], 'bottle': [1, 3], 'mug': [3, 5], 'bowl': [0, 0]}
        shape_names = []
        for obj in ['spoon', 'mug', 'bottle', 'bowl']:
            n = np.random.randint(n_objs[obj][0], n_objs[obj][1] + 1, dtype=int)
            shape_names += random.sample(self.object_lib['names'][obj], k=n)
        random.shuffle(shape_names)
        return shape_names

    def generate_random_objects_in_set(self, obj_set_name, n_objs=1):
        if obj_set_name not in self.object_sets:
            print('wrong object set name, cannot generate objects')
            return []
        return random.sample(self.object_sets[obj_set_name], k=n_objs)

    def read_weight(self, obj):
        # assume object standing, not lying down, need some time for force sensor to react
        time.sleep(2)
        _, _, retFloats, _, _ = self.call_remote('readObjectWeight', int_param=[obj['handle'], obj['forcesensor']])
        lastread = 0
        while not retFloats or abs(retFloats[2] - lastread) > 0.01:  # wait till force sensor becomes stable
            time.sleep(1)
            lastread = retFloats[2]
            _, _, retFloats, _, _ = self.call_remote('readObjectWeight', int_param=[obj['handle'], obj['forcesensor']])
            print(retFloats[2] - lastread)
        return -retFloats[2]  # z axis force downwards

    def estimate_pose2acf(self, estimate_pose, handle, partname):
        try:
            pose = estimate_pose[handle]
            partlabel = self.classes[partname]
            index = np.where(pose[:, -1] == partlabel)[0]
            if isinstance(index, np.ndarray):
                index = index[0]
            return [pose[index, :3] * 1e-2, pose[index, 3:6]]
        except:
            return []

    def gt_obj2acf(self, obj_name, obj_handle, partname):
        pose = self.get_obj_pose(obj_handle)
        kpt = np.dot(pose[:3, :3], self.object_lib[obj_name][partname][0]) + pose[:3, 3]
        axis = np.dot(pose[:3, :3], self.object_lib[obj_name][partname][1])
        return [kpt, axis]

    def get_obj_pose(self, obj_handle):
        sim_ret, pos = sim.simxGetObjectPosition(self.sim_client, obj_handle, -1, sim.simx_opmode_blocking)
        sim_ret, rot = sim.simxGetObjectOrientation(self.sim_client, obj_handle, -1, sim.simx_opmode_blocking)
        trans = np.eye(4)
        trans[0:3, 3] = np.asarray(pos)
        rotm = np.eye(4)
        rotm[:3, :3] = euler2rotm(rot)
        return np.dot(trans, rotm)

    def check_particle_mix(self, container, spoon):
        # use difference between desired position and actual object part keypoint position as constraint
        time.sleep(3)
        container_body_acf = self.gt_obj2acf(container['name'], container['handle'], 'body')
        spoon_head_acf = self.gt_obj2acf(spoon['name'], spoon['handle'], 'head')
        return np.linalg.norm(container_body_acf[0][:2] - spoon_head_acf[0][:2]) < self.stir_dis_threshold

    def show_pose(self, pose):
        pos = pose[:3, -1] + self.reference_frame_offset * np.sum(pose[:3, :3], axis=1)
        rot = rotm2euler(pose[:3, :3])
        sim_ret, obj_handle = sim.simxGetObjectHandle(self.sim_client, 'ReferenceFrame_model', sim.simx_opmode_blocking)
        sim.simxSetObjectPosition(self.sim_client, obj_handle, -1, pos, sim.simx_opmode_blocking)
        sim.simxSetObjectOrientation(self.sim_client, obj_handle, -1, rot, sim.simx_opmode_blocking)
        time.sleep(0.5)

    def set_object_dynamic(self, object_handle, dynamic_bool):
        self.call_remote('setObjectStatic', int_param=[object_handle, 1 - dynamic_bool])

    def set_object_respondable(self, object_handle, respondable_bool):
        self.call_remote('setObjectRespondable', int_param=[object_handle, respondable_bool])

    def visualize_path(self, path, load_file=False, load_filename=''):
        if load_file:
            self.call_remote('visualizePath', int_param=[self.robot_handle, 255, 0, 255, load_file],
                             str_param=[load_filename])
            return 1
        else:
            _, retInts, _, _, _ = self.call_remote('visualizePath',
                                                   int_param=[self.robot_handle, 255, 0, 255, load_file],
                                                   float_param=path)
            if not retInts:
                return 0
            else:
                return retInts[0]

    def clear_visualization_path(self, line_handle):
        self.call_remote('removeLine', int_param=[line_handle])

    def reset_arm(self):
        path = self.path_record
        if not path:
            return
        reverse_path = []
        n_step = len(path) // 7
        for i in range(n_step - 1, -1, -1):  # path are composed of 7 joint angles along the trajectory
            reverse_path += path[7 * i:7 * i + 7]
        self.execute(reverse_path)
        self.path_record = []

    def get_joint_angles(self):
        _, _, retFloats, _, _ = self.call_remote('getJointPositions')
        return retFloats

    def remove_object(self, obj):
        sim.simxRemoveObject(self.sim_client, obj['handle'], sim.simx_opmode_blocking)
        if obj['forcesensor'] != -1:
            sim.simxRemoveObject(self.sim_client, obj['forcesensor'], sim.simx_opmode_blocking)
        if obj['convex_handle'] != -1:
            sim.simxRemoveObject(self.sim_client, obj['convex_handle'], sim.simx_opmode_blocking)
        self.object_list_.remove(obj)

    def update_collision_set(self):
        # add tabletop objects into 'environment0' collection
        handles = [obj['handle'] for obj in self.object_list_]
        self.call_remote('update_collision_set', int_param=handles)

    def attachdetach2gripper(self, obj, attach=1):  # attach if attach==1, detach/attach to force sensor if attach==0
        self.call_remote('attach2Gripper', int_param=[attach, obj['handle'], obj['forcesensor']])
        time.sleep(0.5)

    def close_gripper(self):
        self.call_remote('closeGripper')
        time.sleep(3)
        self.call_remote('holdGripper')

    def open_gripper(self, open_mode=0):
        self.call_remote('openGripper', int_param=[open_mode])

    def check_object_grasped(self, obj_name, obj_handle):
        # joint angle value at full width = -1.6523, half width = 0, close = 0.8897
        if 'Bottle' in obj_name:
            joint_check = [-1.5, 0.4]
        else:
            joint_check = [0.2, 0.88]
        sim_ret, retInt, _, _, _ = self.call_remote('check_grasp', int_param=[obj_handle], float_param=joint_check)
        if sim_ret == 0 and retInt != []:
            return retInt[0]  # 0 miss, 1 grasp
        else:
            print('cannot get result from check_grasp remote call')
            return -1

    def drop_object(self, obj):
        self.reset_arm()
        drop_obj_joints = [deg * math.pi / 180 for deg in [90, 0, 0, 115, 90, 0, 0]]  # joint config over bin
        path = np.linspace(self.read_arm_joints(), drop_obj_joints, num=100).flatten().tolist()
        self.execute(path)
        drop_obj_joints = [deg * math.pi / 180 for deg in [90, 15, 0, 115, 90, 0, 0]]
        path = np.linspace(self.read_arm_joints(), drop_obj_joints, num=100).flatten().tolist()
        self.execute(path)
        self.open_gripper('Bottle' in obj['name'])
        self.attachdetach2gripper(obj, attach=0)
        # self.set_object_respondable(obj['handle'], 1)
        self.set_object_dynamic(obj['handle'], 1)
        time.sleep(4)
        self.remove_object(obj)

    # # archived functions below
    def test_dynamic(self, obj_name):
        _, obj_handle = sim.simxGetObjectHandle(self.sim_client, obj_name, sim.simx_opmode_blocking)
        obj = {'name': obj_name, 'handle': obj_handle}
        self.add_particles(obj)
        self.set_object_dynamic(obj_handle, 1)
        time.sleep(1)
        self.set_object_dynamic(obj_handle, 0)

    # seems not useful as RG2_openCloseJoint is always almost 0
    def check_gripper_closed(self):
        sim_ret, RG2_gripper_handle = sim.simxGetObjectHandle(self.sim_client, 'RG2_openCloseJoint',
                                                              sim.simx_opmode_blocking)
        sim_ret, gripper_joint_position = sim.simxGetJointPosition(self.sim_client, RG2_gripper_handle,
                                                                   sim.simx_opmode_blocking)
        print(gripper_joint_position + 0.0477)
        if gripper_joint_position + 0.0477 < 0.01:
            print('Gripper closed')
            return True

        return False

    def open_gripper_archive(self, width=0.101):
        gripper_motor_velocity = 0.08
        gripper_motor_force = 1000
        sim_ret, RG2_gripper_handle = sim.simxGetObjectHandle(self.sim_client, 'RG2_openCloseJoint',
                                                              sim.simx_opmode_blocking)
        sim_ret, gripper_joint_position = sim.simxGetJointPosition(self.sim_client, RG2_gripper_handle,
                                                                   sim.simx_opmode_blocking)

        if gripper_joint_position + 0.0477 > width:
            print('Gripper is already openned at or greater than the width: ', width)
            return

        sim.simxSetJointForce(self.sim_client, RG2_gripper_handle, gripper_motor_force, sim.simx_opmode_blocking)
        sim.simxSetJointTargetVelocity(self.sim_client, RG2_gripper_handle, gripper_motor_velocity,
                                       sim.simx_opmode_blocking)

        while gripper_joint_position + 0.0477 < width:
            sim_ret, gripper_joint_position = sim.simxGetJointPosition(self.sim_client, RG2_gripper_handle,
                                                                       sim.simx_opmode_blocking)

    def close_gripper_archive(self, width=0.0):
        gripper_motor_velocity = -0.08
        gripper_motor_force = 1000
        sim_ret, RG2_gripper_handle = sim.simxGetObjectHandle(self.sim_client, 'RG2_openCloseJoint',
                                                              sim.simx_opmode_blocking)
        sim_ret, gripper_joint_position = sim.simxGetJointPosition(self.sim_client, RG2_gripper_handle,
                                                                   sim.simx_opmode_blocking)

        if gripper_joint_position + 0.0477 < width:
            print('Gripper is already closed at or smaller than the width: ', width)
            return

        sim.simxSetJointForce(self.sim_client, RG2_gripper_handle, gripper_motor_force, sim.simx_opmode_blocking)
        sim.simxSetJointTargetVelocity(self.sim_client, RG2_gripper_handle, gripper_motor_velocity,
                                       sim.simx_opmode_blocking)
        count = 0

        while gripper_joint_position + 0.0477 > width and count < 30:  # block until the width of the gripper is smaller than desired with or the count is over 50
            sim_ret, gripper_joint_position = sim.simxGetJointPosition(self.sim_client, RG2_gripper_handle,
                                                                       sim.simx_opmode_blocking)

            count += 1
