#!/usr/bin/env python
# -*- coding:utf-8 -*-

# Copyright (c) 2021 SoftBank Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import cv2
import os
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm
from numpy import random

# from yolor.utils.google_utils import attempt_load
from yolor.utils.datasets import create_dataloader, letterbox
from yolor.utils.general import coco80_to_coco91_class, check_dataset, check_file, check_img_size, box_iou, \
    non_max_suppression, scale_coords, xyxy2xywh, xywh2xyxy, clip_coords, set_logging, increment_path, apply_classifier
from yolor.utils.loss import compute_loss
from yolor.utils.metrics import ap_per_class
from yolor.utils.plots import plot_images, output_to_target
from yolor.utils.plots import plot_one_box
from yolor.utils.torch_utils import select_device, time_synchronized, load_classifier

from yolor.models.models import *

from cv_bridge import CvBridge, CvBridgeError
import message_filters
import rospy
import rospkg
from sensor_msgs.msg import Image


def load_classes(path):
    # Loads *.names file at 'path'
    with open(path, 'r') as f:
        names = f.read().split('\n')
    return list(filter(None, names))  # filter removes empty strings (such as last line)


class Recognition(object):
    def __init__(self):
        package_path = rospkg.RosPack().get_path('yolor')

        config_path = rospy.get_param('config_path', package_path + '/configs/config.yaml')
        self.__config = load_cfg_from_cfg_file(config_path)
        self.__config.weights = rospy.get_param('model_path',
                                                package_path + '/' + self.__config.weights)
        self.__config.cfg = rospy.get_param('yolo_cfg_path', package_path + '/' + self.__config.cfg)
        self.__config.data = rospy.get_param('yolo_data_path', package_path + '/' + self.__config.data)
        self.__config.names = rospy.get_param('yolo_name_path', package_path + '/' + self.__config.names)
        

        self.device = select_device(self.__config.device, batch_size=self.__config.batch_size)

        # Load model
        self.model = Darknet(self.__config.cfg).to(self.device)
        try:
            ckpt = torch.load(self.__config.weights, map_location=self.device)  # load checkpoint
            ckpt['model'] = {k: v for k, v in ckpt['model'].items() if self.model.state_dict()[k].numel() == v.numel()}
            self.model.load_state_dict(ckpt['model'], strict=False)
        except BaseException:
            load_darknet_weights(self.model, self.__config.weights)
        imgsz = check_img_size(self.__config.img_size, s=64)  # check img_size

        # Half
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA
        if self.half:
            self.model.half()

        # Configure
        self.model.eval()
        is_coco = self.__config.data.endswith('coco.yaml')  # is COCO dataset
        with open(self.__config.data) as f:
            self.__config.data = yaml.load(f, Loader=yaml.FullLoader)  # model dict
        nc = int(self.__config.data['nc'])  # number of classes
        iouv = torch.linspace(0.5, 0.95, 10).to(self.device)  # iou vector for mAP@0.5:0.95
        niou = iouv.numel()

        self.modelc = load_classifier(name='resnet101', n=2)  # initialize
        # self.modelc.load_state_dict(
        #     torch.load(
        #         'weights/resnet101.pt',
        #         map_location=self.device)['model'])  # load weights
        self.modelc.to(self.device).eval()

        self.__pub_image = rospy.Publisher('~images', Image, queue_size=10)

        self.__cv_bridge = CvBridge()
        rospy.Subscriber("~color", Image, self.__callback_image)

    def __callback_image(self, color):
        try:
            cv_image = self.__cv_bridge.imgmsg_to_cv2(color, 'bgr8')
        except CvBridgeError as e:
            rospy.logerr('Converting Image Error. ' + str(e))
            return
        # Padded resize
        img = letterbox(cv_image, new_shape=self.__config.img_size, auto_size=64)[0]
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        try:
            names = self.model.names if hasattr(self.model, 'names') else self.model.module.names
        except BaseException:
            names = load_classes(self.__config.names)
        coco91class = coco80_to_coco91_class()
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        pred = self.model(img, augment=self.__config.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, self.__config.conf_thres, self.__config.iou_thres)
        
        # Apply Classifier
        pred = apply_classifier(pred, self.modelc, img, cv_image)

        # Process detections
        for det in pred:  # detections per image
            if det is not None and len(det):
                # Rescale boxes from img_size to cv_image size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], cv_image.shape).round()

                # Write results
                for *xyxy, conf, cls in det:
                    label = '%s %.2f' % (names[int(cls)], conf)
                    plot_one_box(xyxy, cv_image, label=label, color=colors[int(cls)], line_thickness=3)

        self.__pub_image.publish(self.__cv_bridge.cv2_to_imgmsg(cv_image, 'bgr8'))


class CfgNode(dict):
    """
    CfgNode represents an internal node in the configuration tree. It's a simple
    dict-like container that allows for attribute-based access to keys.
    """

    def __init__(self, init_dict=None, key_list=None, new_allowed=False):
        # Recursively convert nested dictionaries in init_dict into CfgNodes
        init_dict = {} if init_dict is None else init_dict
        key_list = [] if key_list is None else key_list
        for k, v in init_dict.items():
            if isinstance(v, dict):
                # Convert dict to CfgNode
                init_dict[k] = CfgNode(v, key_list=key_list + [k])
        super(CfgNode, self).__init__(init_dict)

    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        self[name] = value

    def __str__(self):
        def _indent(s_, num_spaces):
            s = s_.split("\n")
            if len(s) == 1:
                return s_
            first = s.pop(0)
            s = [(num_spaces * " ") + line for line in s]
            s = "\n".join(s)
            s = first + "\n" + s
            return s

        r = ""
        s = []
        for k, v in sorted(self.items()):
            seperator = "\n" if isinstance(v, CfgNode) else " "
            attr_str = "{}:{}{}".format(str(k), seperator, str(v))
            attr_str = _indent(attr_str, 2)
            s.append(attr_str)
        r += "\n".join(s)
        return r

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, super(CfgNode, self).__repr__())


def load_cfg_from_cfg_file(file):
    cfg = {}
    assert os.path.isfile(file) and file.endswith(".yaml"), "{} is not a yaml file".format(file)

    with open(file, "r") as f:
        cfg_from_file = yaml.safe_load(f)

    for key, v in cfg_from_file.items():
        cfg[key] = v

    cfg = CfgNode(cfg)
    return cfg


if __name__ == '__main__':
    rospy.init_node('yolor')
    node = Recognition()
    rospy.spin()
