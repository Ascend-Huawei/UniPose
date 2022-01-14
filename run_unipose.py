
"""
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
import cv2
import numpy as np
import argparse
import sys
import time

from acllite_resource import AclLiteResource 
from acllite.acllite_model import AclLiteModel

MODEL_PATH = os.path.join("/home/HwHiAiUser/om/unipose_argmax.om")
# MODEL_PATH = os.path.join("/home/HwHiAiUser/om/unipose_504a3_200dk.om")
print("MODEL_PATH:", MODEL_PATH)

def get_kpts(heat, img_h = 368.0, img_w = 368.0):
    kpts = []
    for m in heat:
        # import matplotlib.pyplot as plt
        # plt.imshow(m)
        # plt.savefig("heat.png")
        h, w = np.unravel_index(m.argmax(), m.shape)
        x = int(w * img_w / m.shape[1])
        y = int(h * img_h / m.shape[0])
        kpts.append([x,y])
    return kpts

def draw(img, kpts):
    for k in kpts:
        cv2.circle(img, k, radius=3, thickness=-1, color=(0, 0, 255))
    cv2.imwrite('outputs/test_out.png', img)

def post_process(heat, img):
    heat = np.moveaxis(heat, 0, -1)
    heat = cv2.resize(heat, dsize=(368, 368), interpolation=cv2.INTER_LINEAR)
    heat = np.moveaxis(heat, -1, 0)
    kpts = get_kpts(heat, img.shape[0], img.shape[1])
    print(kpts)
    draw(img, kpts)

def pre_process(img):
    model_input  = cv2.resize(img, (368,368)).transpose(2, 0, 1)
    mean = 128.0
    std  = 256.0
    model_input= (model_input- mean) / std

    return model_input[None].astype(np.float32).copy()

def main(model_path):
    """main"""
    #initialize acl runtime 
    acl_resource = AclLiteResource()
    acl_resource.init()

    model = AclLiteModel(model_path)

    img = cv2.imread("data/000502550.jpg")
    # img = cv2.resize(img, None, fx=1/3, fy=1/3, interpolation=cv2.INTER_AREA)
    
    st = time.time()
    model_input = pre_process(img)
    print("preprocess time:", time.time() - st); st = time.time()

    output = model.execute([model_input])
    kpts = output[0]
    
    print("inference time:", time.time() - st); st = time.time()

    for x, y in kpts:
        x = np.around(x*img.shape[0]).astype(int)
        y = np.around(y*img.shape[1]).astype(int)
        cv2.circle(img, [y,x], radius=3, thickness=-1, color=(0, 0, 255))
    cv2.imwrite('outputs/000502550.png', img)
    print("post_process (draw) time:", time.time() - st); st = time.time()
    

if __name__ == '__main__':   
    description = 'Load a model for human pose estimation'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--model', type=str, default=MODEL_PATH)

    args = parser.parse_args()
    
    main(args.model)
