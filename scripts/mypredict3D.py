import click
from jarvis.utils.paramClasses import Predict3DParams
import os
import csv
import itertools
import numpy as np
import torch
import cv2
import json
import itertools
from joblib import Parallel, delayed
from tqdm import tqdm
import time
from ruamel.yaml import YAML

from jarvis.utils.reprojection import ReprojectionTool, load_reprojection_tools
from jarvis.utils.reprojection import get_repro_tool
from jarvis.config.project_manager import ProjectManager
from jarvis.prediction.jarvis3D import JarvisPredictor3D

import pandas as pd

def mypredict3d(params, trials_frames):
    #Load project and config
    project = ProjectManager()
    if not project.load(params.project_name):
        print (f'{CLIColors.FAIL}Could not load project: {project_name}! '
                    f'Aborting....{CLIColors.ENDC}')
        return
    cfg = project.cfg

    jarvisPredictor = JarvisPredictor3D(cfg, params.weights_center_detect,
                params.weights_hybridnet, params.trt_mode)
    reproTool = get_repro_tool(cfg, params.dataset_name)

    params.output_dir = os.path.join(params.recording_path, 'jarvis_prediction', params.project_name,
                f'Predictions_3D_{time.strftime("%Y%m%d-%H%M%S")}')

    os.makedirs(params.output_dir, exist_ok = True)
    create_info_file(params)
    #create openCV video read streams
    video_paths = get_video_paths(
                params.recording_path, reproTool)
    caps, img_size = create_video_reader(params, reproTool,
                video_paths)

    csvfile = open(os.path.join(params.output_dir, 'data3D.csv'), 'w',
                newline='')
    writer = csv.writer(csvfile, delimiter=',',
                    quotechar='"', quoting=csv.QUOTE_MINIMAL)
    #if keypoint names are defined, add header to csvs
    if (len(cfg.KEYPOINT_NAMES) == cfg.KEYPOINTDETECT.NUM_JOINTS):
        create_header(writer, cfg)

    imgs_orig = np.zeros((len(caps), img_size[1],
                img_size[0], 3)).astype(np.uint8)

    for trial_idx in tqdm(range(trials_frames.shape[0])):
        frame_start, frame_end = trials_frames[trial_idx]
        
        ## seek
        Parallel(n_jobs=17, require='sharedmem')(delayed(seek)(cap, frame_start) for cap in caps)

        for frame_num in range(frame_start, frame_end+1):
            #load a batch of images from all cameras in parallel using joblib
            Parallel(n_jobs=17, require='sharedmem')(delayed(read_images)
                        (cap, slice, imgs_orig) for slice, cap in enumerate(caps))
            imgs = torch.from_numpy(
                    imgs_orig).cuda().float().permute(0,3,1,2)[:, [2, 1, 0]]/255.

            points3D_net, confidences = jarvisPredictor(imgs,
                        reproTool.cameraMatrices.cuda(),
                        reproTool.intrinsicMatrices.cuda(),
                        reproTool.distortionCoefficients.cuda())

            if points3D_net != None:
                row = [frame_num]
                for point, conf in zip(points3D_net.squeeze(), confidences.squeeze().cpu().numpy()):
                    row = row + point.tolist() + [conf]
                writer.writerow(row)
            else:
                row = [frame_num]
                for i in range(cfg.KEYPOINTDETECT.NUM_JOINTS*4):
                    row = row + ['NaN']
                writer.writerow(row)

    for cap in caps:
        cap.release()
    csvfile.close()
    return params.output_dir


def create_video_reader(params, reproTool, video_paths):
    caps = []
    img_size = [0,0]
    for i,path in enumerate(video_paths):
        cap = cv2.VideoCapture(path)
        img_size_new = [int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                     int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))]
        assert (img_size == [0,0] or img_size == img_size_new), \
                    "All videos need to have the same resolution"
        img_size = img_size_new
        caps.append(cap)

    return caps, img_size


def get_video_paths(recording_path, reproTool):
    videos = os.listdir(recording_path)
    video_paths = []
    for i, camera in enumerate(reproTool.cameras):
        for video in videos:
            if camera == video.split('.')[0]:
                video_paths.append(os.path.join(recording_path, video))
        assert (len(video_paths) == i+1), \
                    "Missing Recording for camera " + camera
    return video_paths


def seek(cap, frame_num):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)


def read_images(cap, slice, imgs):
    ret, img = cap.read()
    imgs[slice] = img.astype(np.uint8)


def create_header(writer, cfg):
    joints = list(itertools.chain.from_iterable(itertools.repeat(x, 4)
                for x in cfg.KEYPOINT_NAMES))
    coords = ['x','y','z', 'confidence']*len(cfg.KEYPOINT_NAMES)
    joints.insert(0, 'frame')
    coords.insert(0, 'frame')
    writer.writerow(joints)
    writer.writerow(coords)


def create_info_file(params):
    with open(os.path.join(params.output_dir, 'info.yaml'), 'w') as file:
        yaml=YAML()
        yaml.dump({'recording_path': params.recording_path,
                    'dataset_name': params.dataset_name}, file)


@click.command()
@click.option('--weights_center_detect', default = 'latest',
            help = 'CenterDetect weights to load for prediction. You have to '
            'specify the path to a specific \'.pth\' file')
@click.option('--weights_hybridnet', default = 'latest',
            help = 'HybridNet weights to load for prediction. You have to '
            'specify the path to a specific \'.pth\' file')
@click.option('--dataset_name', default = None,
            help = 'If your dataset contains multiple calibrations, specify '
            'which one you want to use by giving the name of the dataset it '
            'belongs to. You can also specify a path to any folder containing '
            'valid calibrations for your recording.')
@click.argument('project_name')
@click.argument('recording_path')
def predict3D(project_name, recording_path, weights_center_detect,
            weights_hybridnet, dataset_name):
    """
    Predict 3D poses on a multi-camera recording.
    """

    trial_start_end_filename = os.path.join(recording_path, "trial_start_end.csv")
    trial_df = pd.read_csv(trial_start_end_filename)
    trials_frames = trial_df.to_numpy()

    params = Predict3DParams(project_name, recording_path)
    params.weights_center_detect = weights_center_detect
    params.weights_hybridnet = weights_hybridnet
    params.dataset_name = dataset_name
    return mypredict3d(params, trials_frames)


if __name__ == '__main__':
    output_dir = predict3D()
    print(output_dir)