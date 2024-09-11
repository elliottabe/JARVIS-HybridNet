import os
import csv
import itertools
import numpy as np
import torch
import cv2
from tqdm import tqdm
import time
from ruamel.yaml import YAML

from jarvis.prediction.jarvis2D import JarvisPredictor2D
from jarvis.config.project_manager import ProjectManager
import pandas as pd

def create_info_file(params):
    with open(os.path.join(params.output_dir, 'info.yaml'), 'w') as file:
        yaml=YAML()
        yaml.dump({'recording_path': params.recording_path}, file)


def mypredict2D(params, trials_frames):
    project = ProjectManager()
    if not project.load(params.project_name):
        print (f'{CLIColors.FAIL}Could not load project: {project_name}! '
                    f'Aborting....{CLIColors.ENDC}')
        return
    cfg = project.cfg

    params.output_dir = os.path.join(project.recording_path, 'jarvis_prediction', params.project_name, 'predictions2D',
                f'Predictions_2D_{time.strftime("%Y%m%d-%H%M%S")}')
    
    os.makedirs(params.output_dir, exist_ok = True)
    create_info_file(params)

    jarvisPredictor = JarvisPredictor2D(cfg, params.weights_center_detect,
                params.weights_keypoint_detect, params.trt_mode)
    video_file = os.path.join(project.recording_path, "Cam710038.mp4")

    cap = cv2.VideoCapture(video_file)
    img_size  = [int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))]

    csv_filename = 'data2D.csv'
    csvfile = open(os.path.join(params.output_dir, csv_filename), 'w',
                newline='')
    writer = csv.writer(csvfile, delimiter=',',
                    quotechar='"', quoting=csv.QUOTE_MINIMAL)

    #if keypoint names are defined, add header to csvs
    if (len(cfg.KEYPOINT_NAMES) == cfg.KEYPOINTDETECT.NUM_JOINTS):
        create_header(writer, cfg)

    for trial_idx in tqdm(range(trials_frames.shape[0])):
        frame_start, frame_end = trials_frames[trial_idx]
        seek(cap, frame_start)

        for frame_num in range(frame_start, frame_end+1):
            ret, img_orig = cap.read()
            img = torch.from_numpy(
                    img_orig).cuda().float().permute(2,0,1)[[2, 1, 0]]/255.

            points2D, confidences = jarvisPredictor(img.unsqueeze(0))

            if points2D != None:
                points2D = points2D.cpu().numpy()
                confidences = confidences.cpu().numpy()
                row = [frame_num]
                for i,point in enumerate(points2D):
                    row = row + point.tolist() + [confidences[i]]
                writer.writerow(row)

            else:
                row = [frame_num]
                for i in range(cfg.KEYPOINTDETECT.NUM_JOINTS*3):
                    row = row + ['NaN']
                writer.writerow(row)

    cap.release()
    csv_filename.close()
    return params.output_dir

def seek(cap, frame_num):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)

def create_header(writer, cfg):
    joints = list(itertools.chain.from_iterable(itertools.repeat(x, 3)
                for x in cfg.KEYPOINT_NAMES))
    coords = ['x','y','confidence']*len(cfg.KEYPOINT_NAMES)
    joints.insert(0, 'frame')
    coords.insert(0, 'frame')    
    writer.writerow(joints)
    writer.writerow(coords)


import click
from jarvis.prediction.predict2D import predict2D as predict2D_funct
from jarvis.utils.paramClasses import Predict2DParams

@click.command()
@click.option('--weights_center_detect', default = 'latest',
            help = 'CenterDetect weights to load for prediction. You have to '
            'specify the path to a specific \'.pth\' file')
@click.option('--weights_keypoint_detect', default = 'latest',
            help = 'KeypointDetect weights to load for prediction. You have to '
            'specify the path to a specific \'.pth\' file')
@click.argument('project_name')
@click.argument('video_path')
def predict2D(project_name, video_path, weights_center_detect,
            weights_keypoint_detect, frame_start, number_frames):
    """
    Predict 2D poses on a single video.
    """
    ## get recording path as output file 
    trial_start_end_filename = os.path.join(video_path, "trial_start_end.csv")
    trial_df = pd.read_csv(trial_start_end_filename)
    trials_frames = trial_df.to_numpy()

    params = Predict2DParams(project_name, video_path)
    params.weights_center_detect = weights_center_detect
    params.weights_keypoint_detect = weights_keypoint_detect
    return mypredict2D(params, trials_frames)


if __name__ == '__main__':
    output_dir = predict2D()
    print(output_dir)