"""
JARVIS-MoCap (https://jarvis-mocap.github.io/jarvis-docs)
Copyright (c) 2022 Timo Hueser.
https://github.com/JARVIS-MoCap/JARVIS-HybridNet
Licensed under GNU Lesser General Public License v2.1
"""

import os,sys,inspect
import numpy as np
import cv2
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import jarvis.utils.numpy2_compat  # noqa: F401  (restore np aliases for imgaug on numpy>=2)
import imgaug.augmenters as iaa
from imgaug.augmentables import (Keypoint, KeypointsOnImage,
                                BoundingBox, BoundingBoxesOnImage)
from imgaug.augmentables.segmaps import SegmentationMapsOnImage

current_dir = os.path.dirname(os.path.abspath(
            inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, parent_dir)

from jarvis.dataset.datasetBase import BaseDataset


class Dataset2D(BaseDataset):
    """
    Dataset Class to load 2D datasets in the HybridNet dataset format,
    inherits from BaseDataset class. See HERE for more details.

    :param cfg: handle of the global configuration
    :param set: specifies wether to load training ('train') or validation
                ('val') split. Augmentation will only be applied to
                training split.
    :type set: string
    :param mode: specifies wether center of mass ('center') or keypoint
                 annotations ('keypoints') will be loaded.
    :type mode: string
    """
    def __init__(self, cfg, set='train', mode = 'CenterDetect', **kwargs):
        dataset_name = cfg.DATASET.DATASET_2D
        super().__init__(cfg, dataset_name,set, **kwargs)
        self.mode = mode
        assert cfg.KEYPOINTDETECT.BOUNDING_BOX_SIZE % 64 == 0, \
                    "Bounding Box size has to be divisible by 64!"

        self.instance_mask_input = (
            mode == 'KeypointDetect'
            and getattr(cfg.KEYPOINTDETECT, 'INSTANCE_MASK_INPUT', False))

        # When INSTANCE_MASK_INPUT is on, KeypointDetect iterates per annotation
        # (so each fly in a multi-fly frame is seen independently) rather than
        # per image. Build a flat [(image_id, ann_idx)] index once at init.
        self.ann_index = None
        if self.instance_mask_input:
            self.ann_index = []
            for img_id in self.image_ids:
                anns = self.imgToAnns.get(img_id, [])
                for j in range(len(anns)):
                    self.ann_index.append((img_id, j))

        img = self._load_image(0)
        width, height = img.shape[1], img.shape[0]
        cfg.DATASET.IMAGE_SIZE = [width,height]

        if self.mode == 'CenterDetect':
            cfg.CENTERDETECT.NUM_JOINTS = 1
            img = self._load_image(0)
            self.width, self.height = img.shape[1], img.shape[0]

            self.heatmap_generators = []
            output_sizes = [[int(self.cfg.CENTERDETECT.IMAGE_SIZE/4),
                             int(self.cfg.CENTERDETECT.IMAGE_SIZE/4)],
                            [int(self.cfg.CENTERDETECT.IMAGE_SIZE/2),
                             int(self.cfg.CENTERDETECT.IMAGE_SIZE/2)]]
            for output_size in output_sizes:
                self.heatmap_generators.append(HeatmapGenerator(
                                [self.cfg.CENTERDETECT.IMAGE_SIZE,
                                self.cfg.CENTERDETECT.IMAGE_SIZE], output_size,
                                cfg.CENTERDETECT.NUM_JOINTS, sigma = -2))

        elif self.mode == 'KeypointDetect':
            self.heatmap_generators = []
            output_sizes = [int(cfg.KEYPOINTDETECT.BOUNDING_BOX_SIZE/4),
                            int(cfg.KEYPOINTDETECT.BOUNDING_BOX_SIZE/2)]
            for output_size in output_sizes:
                self.heatmap_generators.append(HeatmapGenerator(
                            [cfg.KEYPOINTDETECT.BOUNDING_BOX_SIZE,
                            cfg.KEYPOINTDETECT.BOUNDING_BOX_SIZE] ,
                            [output_size,output_size], self.num_keypoints[0]))

        self._build_augpipe()
        self.transform = transforms.Compose([Normalizer(mean=cfg.DATASET.MEAN,
                                                        std=cfg.DATASET.STD)])


    def _build_augpipe(self):
        augmentors = []
        if self.mode == 'CenterDetect':
            img = self._load_image(0)
            width, height = img.shape[1], img.shape[0]
            scale = 1./self.cfg.CENTERDETECT.IMAGE_SIZE
            self.scale_width = float(width)/self.cfg.CENTERDETECT.IMAGE_SIZE
            self.scale_height = float(height)/self.cfg.CENTERDETECT.IMAGE_SIZE

            augmentors.append(iaa.Resize(self.cfg.CENTERDETECT.IMAGE_SIZE,
                        interpolation='linear'))

        if not (self.mode == 'CenterDetect' and self.set_name == 'val'):
            cfg = self.cfg.AUGMENTATION
            if cfg.COLOR_MANIPULATION.ENABLED:
                cman = cfg.COLOR_MANIPULATION
                augmentors.append(
                    iaa.Sometimes(cman.GAUSSIAN_BLUR.PROBABILITY,
                    iaa.GaussianBlur(sigma=cman.GAUSSIAN_BLUR.SIGMA)))
                augmentors.append(
                    iaa.AdditiveGaussianNoise(
                    loc = 0, scale = cman.GAUSSIAN_NOISE.SCALE,
                    per_channel = cman.GAUSSIAN_NOISE.PER_CHANNEL_PROBABILITY))
                augmentors.append(
                    iaa.Sometimes(cman.LINEAR_CONTRAST.PROBABILITY,
                    iaa.LinearContrast(cman.LINEAR_CONTRAST.SCALE)))
                augmentors.append(
                    iaa.Sometimes(cman.MULTIPLY.PROBABILITY,
                    iaa.Multiply(cman.MULTIPLY.SCALE)))
                augmentors.append(
                    iaa.Sometimes(cman.PER_CHANNEL_MULTIPLY.PROBABILITY,
                    iaa.Multiply(cman.PER_CHANNEL_MULTIPLY.SCALE,
                    per_channel =
                    cman.PER_CHANNEL_MULTIPLY.PER_CHANNEL_PROBABILITY)))
            if self.mode == 'KeypointDetect':
                augmentors.append(
                    iaa.Fliplr(cfg.MIRROR.PROBABILITY))
            augmentors.append(
                iaa.Sometimes(cfg.AFFINE_TRANSFORM.PROBABILITY,
                iaa.Affine(rotate=cfg.AFFINE_TRANSFORM.ROTATION_RANGE,
                           scale=cfg.AFFINE_TRANSFORM.SCALE_RANGE)))

        self.augpipe = iaa.Sequential(augmentors, random_order = False)


    def __getitem__(self, idx):
        if self.mode == 'CenterDetect':
            return self._get_item_center(idx)
        else:
            return self._get_item_keypoints(idx)


    def _get_item_center(self,idx):
        img = self._load_image(idx)
        bboxs, keypoints = self._load_annotations(idx)

        # Collect centers and sizes for ALL valid animals
        centers = []
        animal_size = 0
        for b in bboxs:
            if b[4] != -1:
                cx = (b[0] + b[2]) / 2
                cy = (b[1] + b[3]) / 2
                centers.append([cx, cy, 1])
                animal_size = max(animal_size,
                                  max(b[3] - b[1], b[2] - b[0]))

        has_valid = len(centers) > 0
        if not has_valid:
            centers = [[0.0, 0.0, 1.0]]
        centers = np.array(centers)

        # Augment all centers together
        keypoints_iaa = KeypointsOnImage(
                    [Keypoint(x=c[0], y=c[1]) for c in centers],
                    shape=(self.height,self.width,3))
        img, keypoints_aug = self.augpipe(image=img, keypoints = keypoints_iaa)
        for i, kp in enumerate(keypoints_aug.keypoints):
            centers[i][0] = kp.x
            centers[i][1] = kp.y

        # Shape (num_animals, 1_joint, 3): outer loop in HeatmapGenerator
        # iterates over "persons", each with 1 joint on channel 0.
        # np.maximum overlay ensures overlapping Gaussians combine correctly.
        joints = np.zeros((len(centers), 1, 3))
        for i, c in enumerate(centers):
            joints[i, 0, :] = c

        joints_list = [[],[]]
        if has_valid:
            joints_list = [joints.copy() for _ in range(2)]
        target_list = list()
        for scale_id in range(2):
            target_t = self.heatmap_generators[scale_id](joints_list[scale_id],
                        animal_size)
            target_list.append(target_t.astype(np.float32))
        # Only return the first center in sample[2] for batch-collation
        # (calculate_accuracy is single-peak; heatmap already encodes all peaks).
        sample = [img, target_list, centers[:1]]
        return self.transform(sample)


    def _get_item_keypoints(self, idx):
        if self.instance_mask_input:
            image_id, target_ann_idx = self.ann_index[idx]
            img = self._load_image(image_id, is_id=True)
            bboxs, keypoints = self._load_annotations(image_id, is_id=True)
            mask_bundle = self._load_instance_masks(image_id, is_id=True)
        else:
            image_id = None
            target_ann_idx = 0
            img = self._load_image(idx)
            bboxs, keypoints = self._load_annotations(idx)
            mask_bundle = None

        bbox = bboxs[target_ann_idx]
        animal_size = np.max([bbox[3] - bbox[1], bbox[2] - bbox[0]])
        bbox_hw = int(self.cfg.KEYPOINTDETECT.BOUNDING_BOX_SIZE / 2)
        center_y = min(max(bbox_hw, int((bbox[1] + int(bbox[3])) / 2)),
                    img.shape[0] - bbox_hw)
        center_x = min(max(bbox_hw, int((bbox[0] + int(bbox[2])) / 2)),
                    img.shape[1] - bbox_hw)

        # Build distractor mask and target mask in full-image coords before
        # cropping, so we can replace distractor pixels with a gray fill in
        # the crop and feed the target mask as a 4th input channel.
        target_mask_crop = None
        if self.instance_mask_input and mask_bundle is not None:
            full_h, full_w = img.shape[:2]
            masks = mask_bundle.get('masks')
            matched = mask_bundle.get('matched')
            extra_masks = mask_bundle.get('extra_masks')

            distractor_full = np.zeros((full_h, full_w), dtype=bool)
            if masks is not None and masks.shape[0] > 0:
                for j in range(masks.shape[0]):
                    if j == target_ann_idx:
                        continue
                    if matched is None or matched[j]:
                        distractor_full |= masks[j]
            if extra_masks is not None and extra_masks.shape[0] > 0:
                for j in range(extra_masks.shape[0]):
                    distractor_full |= extra_masks[j]

            # Exclude any distractor pixels that overlap the target mask so we
            # don't accidentally gray out part of the target fly.
            if (masks is not None and masks.shape[0] > target_ann_idx
                    and (matched is None or matched[target_ann_idx])):
                distractor_full &= ~masks[target_ann_idx]

            distractor_crop = distractor_full[center_y - bbox_hw:center_y + bbox_hw,
                                              center_x - bbox_hw:center_x + bbox_hw]
            # Replace distractor pixels with the crop's mean color so the net
            # sees "no other fly here" and must rely on RGB + the target mask
            # channel to localize this fly's keypoints.
            if distractor_crop.any():
                # Work on the full image crop now.
                img_crop = img[center_y - bbox_hw:center_y + bbox_hw,
                               center_x - bbox_hw:center_x + bbox_hw, :].copy()
                mean_color = img_crop.mean(axis=(0, 1))
                img_crop[distractor_crop] = mean_color
                img = img_crop
            else:
                img = img[center_y - bbox_hw:center_y + bbox_hw,
                          center_x - bbox_hw:center_x + bbox_hw, :]

            if (masks is not None and masks.shape[0] > target_ann_idx
                    and (matched is None or matched[target_ann_idx])):
                target_mask_crop = masks[target_ann_idx,
                                         center_y - bbox_hw:center_y + bbox_hw,
                                         center_x - bbox_hw:center_x + bbox_hw]
                target_mask_crop = target_mask_crop.astype(np.uint8)
            else:
                target_mask_crop = np.zeros(
                    (self.cfg.KEYPOINTDETECT.BOUNDING_BOX_SIZE,
                     self.cfg.KEYPOINTDETECT.BOUNDING_BOX_SIZE), dtype=np.uint8)
        else:
            img = img[center_y - bbox_hw:center_y + bbox_hw,
                      center_x - bbox_hw:center_x + bbox_hw, :]

        kp_row = keypoints[target_ann_idx].copy()
        for i in range(0, len(kp_row), 3):
            kp_row[i] += -center_x + bbox_hw
            kp_row[i + 1] += -center_y + bbox_hw

        if self.set_name == 'train':
            keypoints_iaa = KeypointsOnImage([
                        Keypoint(x=kp_row[i], y=kp_row[i + 1])
                        for i in range(0, len(kp_row), 3)],
                        shape=(self.cfg.KEYPOINTDETECT.BOUNDING_BOX_SIZE,
                        self.cfg.KEYPOINTDETECT.BOUNDING_BOX_SIZE, 3))
            if target_mask_crop is not None:
                segmap = SegmentationMapsOnImage(
                    target_mask_crop, shape=img.shape)
                img, keypoints_aug, segmap_aug = self.augpipe(
                    image=img, keypoints=keypoints_iaa,
                    segmentation_maps=segmap)
                target_mask_crop = segmap_aug.get_arr().astype(np.uint8)
            else:
                img, keypoints_aug = self.augpipe(image=img,
                            keypoints=keypoints_iaa)
            for i, point in enumerate(keypoints_aug.keypoints):
                kp_row[i * 3] = point.x
                kp_row[i * 3 + 1] = point.y

        kp_out = kp_row.copy()
        for i, keypoint in enumerate(kp_row.reshape((-1, 3))):
            if (keypoint[0] < 0 or keypoint[1] < 0
                    or keypoint[0] >= self.cfg.KEYPOINTDETECT.BOUNDING_BOX_SIZE
                    or keypoint[1] >= self.cfg.KEYPOINTDETECT.BOUNDING_BOX_SIZE):
                kp_out[i * 3:i * 3 + 2] = 0
        kp_out = kp_out.reshape(1, -1)

        joints = np.zeros((1, self.num_keypoints[0], 3))
        joints[0, :self.num_keypoints[0], :3] = kp_out[0].reshape([-1, 3])
        joints_list = [joints.copy() for _ in range(2)]
        target_list = list()
        for scale_id in range(2):
            target_t = self.heatmap_generators[scale_id](joints_list[scale_id],
                        animal_size)
            target_list.append(target_t.astype(np.float32))

        if self.instance_mask_input:
            mask_channel = (target_mask_crop.astype(np.float32)
                            if target_mask_crop is not None
                            else np.zeros(img.shape[:2], dtype=np.float32))
            img = np.concatenate([img, mask_channel[..., None]], axis=-1)

        sample = [img, target_list, kp_out]
        sample = self.transform(sample)
        return sample

    def __len__(self):
        if self.instance_mask_input and self.ann_index is not None:
            return len(self.ann_index)
        return len(self.image_ids)


    def get_dataset_config(self):
        """
        Get the recommended configuration for the 2D Dataset. Recommendations
        are computed by analyzing the trainingset, if it is not representative
        of the data you plan to analyze, the parameters might need to be
        adjusted manually
        """
        bboxs = []
        for id in self.image_ids:
            bbox, _ = self._load_annotations(id)
            if len(bbox) != 0:
                bboxs.append(bbox)
        bboxs = np.array(bboxs)
        x_sizes = bboxs[:,0,2]-bboxs[:,0,0]
        y_sizes = bboxs[:,0,3]-bboxs[:,0,1]
        # plt.hist(x_sizes, bins='auto')
        # plt.hist(y_sizes, bins='auto')
        # plt.show()
        bbox_min_size = np.max([np.percentile(x_sizes,98),
                    np.percentile(y_sizes,98)])
        ind = np.argmax(x_sizes)
        file_name = self.imgs[self.image_ids[ind]]['file_name']
        path = os.path.join(self.root_dir, self.set_name,file_name)

        final_bbox_suggestion = int(np.ceil((bbox_min_size*1.20)/64)*64)
        return final_bbox_suggestion


    def visualize_sample(self, idx):
        sample = self.__getitem__(idx)
        img = (sample[0]*self.cfg.DATASET.STD+self.cfg.DATASET.MEAN)
        img = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_RGB2BGR)
        heatmaps = sample[1]
        img = cv2.resize(img*255, (heatmaps[1][0].shape[1],
                                   heatmaps[1][0].shape[0])).astype(np.uint8)
        colored_heatmap = cv2.applyColorMap(heatmaps[1][0].astype(np.uint8),
                    cv2.COLORMAP_JET)
        for i in range(1,heatmaps[1].shape[0]):
            colored_heatmap = colored_heatmap + cv2.applyColorMap(
                        heatmaps[1][i].astype(np.uint8), cv2.COLORMAP_JET)
        img = cv2.addWeighted(img,1.0,colored_heatmap,0.4,0)
        img = cv2.resize(img, (640,512))
        cv2.imshow('frame', img)
        cv2.waitKey(0)


class Normalizer(object):
    def __init__(self, mean, std):
        self.mean = np.array([[mean]])
        self.std = np.array([[std]])

    def __call__(self, sample):
        image, heatmaps = sample[0], sample[1]
        keypoints = sample[2]
        image = image.astype(np.float32)
        # Normalize RGB channels; pass the optional mask channel through as
        # a {0,1} float so the first conv learns to fuse it with RGB.
        rgb = (image[..., :3] - self.mean) / self.std
        if image.shape[-1] > 3:
            image = np.concatenate([rgb, image[..., 3:]], axis=-1)
        else:
            image = rgb
        return [image, heatmaps, keypoints]


class HeatmapGenerator():
    def __init__(self, original_res, output_res, num_joints, sigma=-1):
        self.output_res = output_res
        self.num_joints = num_joints
        self.scale_factor = float(output_res[0])/float(original_res[0])
        if sigma == -1:
            sigma = 1.5*self.output_res[0]/64
            self.fact = 1.0
        elif sigma == -2:
            sigma = 1*self.output_res[0]/64
            self.fact = 0.5
        self.sigma = sigma
        size = 6*sigma + 3
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0, y0 = 3*sigma + 1, 3*sigma + 1
        self.g = 255.0*np.exp(- ((x-x0)**2 + (y-y0)**2) / (2*sigma**2))

    def __call__(self, joints, size):
        hms = np.zeros((self.num_joints, self.output_res[0],
                    self.output_res[1]), dtype=np.float32)

        #sigma = self.fact*size/64.0
        sigma = self.sigma
        size = 6*sigma + 3
        xx = np.arange(0, size, 1, float)
        yy = xx[:, np.newaxis]
        x0, y0 = 3*sigma + 1, 3*sigma + 1
        self.g = 255.0*np.exp(- ((xx-x0)**2 + (yy-y0)**2) / (2*sigma**2))
        for p in joints:
            for idx, pt in enumerate(p):
                if pt[0] != 0 or pt[1] != 0:
                    x, y = (int(pt[0]*self.scale_factor),
                           int(pt[1]*self.scale_factor))
                    if x < 0 or y < 0 or \
                       x >= self.output_res[1] or y >= self.output_res[0]:
                        continue

                    ul = (int(np.round(x - 3 * sigma - 1)),
                         int(np.round(y - 3 * sigma - 1)))
                    br = (int(np.round(x + 3 * sigma + 2)),
                         int(np.round(y + 3 * sigma + 2)))

                    a = max(0, -ul[1])
                    b = min(br[1], self.output_res[0]) - ul[1]
                    c = max(0, -ul[0])
                    d = min(br[0], self.output_res[1]) - ul[0]

                    aa = max(0, ul[1])
                    bb = min(br[1], self.output_res[0])
                    cc = max(0, ul[0])
                    dd = min(br[0], self.output_res[1])

                    hms[idx, aa:bb, cc:dd] = np.maximum(
                        hms[idx, aa:bb, cc:dd], self.g[a:b, c:d])
        return hms


if __name__ == "__main__":
    from jarvis.config.project_manager import ProjectManager
    project = ProjectManager()
    project.load('Rat_Full')
    cfg = project.get_cfg()
    print (cfg.DATASET.DATASET_2D)

    training_set = Dataset2D(cfg = cfg, set='val', mode='KeypointDetect')#, cameras_to_use = ['Camera_T', 'Camera_B'])
    print (len(training_set.image_ids))
    for i in range(0,len(training_set.image_ids),1):
        training_set.visualize_sample(i)
        #print (i)
        #training_set.__getitem__(i)
