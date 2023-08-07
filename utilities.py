# Import Libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import cv2
import numpy as np

from scipy import ndimage
from skimage.measure import label, regionprops

sys.path.insert(0, "./yolov7")
from yolov7.models.experimental import Ensemble
from yolov7.models.common import Conv
from yolov7.utils.datasets import letterbox
from yolov7.utils.general import check_img_size, non_max_suppression, scale_coords

sys.path.insert(0, "./deeplab")
from deeplab.deeplabv2 import DeepLabV2
from deeplab.msc import MSC

# ================== General Utilities ==================
"""
Generates a transparent RGB mask based on a binary mask
"""
def create_overlay(mask):
    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)
    r[mask == 1], g[mask == 1], b[mask == 1] = [0, 255, 0]
    
    overlay = np.stack([r, g, b], axis=2)
    
    return overlay

"""
Generates a white RGB mask based on a binary mask 
"""
def white_mask(image):
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    r[image == 1], g[image == 1], b[image == 1] = [255, 255, 255]
    mask = np.stack([r, g, b], axis=2)

    return mask

"""
Does the following pre-processing steps for the image:
1. resize the image to a width of 640 while preserving the overall aspect ratio
2. normalize by substracting the original RGB values
3. convert the image to torch format for passing it to deeplab model
"""
def preprocessing(image):
    height, width, _ = image.shape

    # Resize
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    scale = width / 640
    image = cv2.resize(image, dsize=(round(width / scale), round(height / scale)), interpolation=cv2.INTER_LINEAR)
    raw_image = image.astype(np.uint8)

    # Subtract mean values
    image = image.astype(np.float32)
    image -= np.array(
        [
            float(104.008),
            float(116.669),
            float(122.675),
        ]
    )

    # Convert to torch.Tensor and add "batch" axis
    image = torch.from_numpy(image.transpose(2, 0, 1)).float().unsqueeze(0)

    return image, raw_image

# ================== Image Processing Techniques / Water Hyacinth Detection ==================
def infer_thresholding(segmented_image):
    hsv_img = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv_img, np.array([50, 100, 40]), np.array([90, 150, 160]))

    # Noise removal using Morphological open operation
    kernel = np.ones((3, 3), np.uint8)
    dilate = cv2.dilate(mask, kernel, iterations=3)
    morphClose = cv2.morphologyEx(dilate, cv2.MORPH_CLOSE, kernel, iterations=5)
    morphOpen = cv2.morphologyEx(morphClose, cv2.MORPH_OPEN, kernel, iterations=2)

    labels = regionprops(label(morphOpen))

    # retrieve small objects and medium + large objects
    small_coords = []
    medium_large_coords = []
    for l in labels:
        coords = np.array([[[b, a] for a, b in l.coords]])
        if(l.area <= 32 * 32):
            small_coords.append((coords[0], l.area))
        else: # if area > 32*32
            medium_large_coords.append((coords[0], l.area))

    return small_coords, medium_large_coords
    

# ================== YOLO v7 Utilities / Garbage Detection ==================
""" 
Load the YoLo v7 model from the trained weights
"""
def load_yolov7(device):
    model = Ensemble()

    weight = torch.load('./model/yolo.pt',
                        map_location=device)  # load FP32 model
    model.append(weight['ema' if weight.get('ema')
                 else 'model'].float().fuse().eval())

    # Compatibility updates
    for m in model.modules():
        if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True  # pytorch 1.7.0 compatibility
        elif type(m) is nn.Upsample:
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility
        elif type(m) is Conv:
            m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility

    if len(model) == 1:
        return model[-1]  # return model
    else:
        print('Ensemble created with %s\n' % weights)
        for k in ['names', 'stride']:
            setattr(model, k, getattr(model[-1], k))
        return model  # return ensemble

""" 
Infer the YoLo v7 model by passing an input image 
"""
def infer_yolov7(segmented_img, device):
    model = load_yolov7(device)
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(640, s=stride)  # check img_size

    # Run inference
    if device != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(
            next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    # Get names and colors
    # names = model.module.names if hasattr(model, 'module') else model.names

    img = letterbox(segmented_img, imgsz, stride)[0]
    
    # Convert
    img = img.transpose(2, 0, 1)  # to 3xheightxwidth
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img / 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Warmup
    if device != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
        old_img_b = img.shape[0]
        old_img_h = img.shape[2]
        old_img_w = img.shape[3]
        for i in range(3):
            model(img)[0]

    # Inference
    with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
        pred = model(img)[0]

    # Apply NMS
    pred = non_max_suppression(pred)

    bboxes = []
    # scores = []
    # Process detections
    for det in pred:  # detections per image
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(
                img.shape[2:], det[:, :4], segmented_img.shape).round()

            for *xyxy, conf, _ in reversed(det):
                bboxes.append(torch.tensor(xyxy).view(1, 4)[0].numpy())
                # scores.append(conf.cpu().numpy())

    small_coords = []
    medium_large_coords = []

    for x1, y1, x2, y2 in bboxes:
        width = x2 - x1
        height = y2 - y1

        if(height * width <=- 32 * 32):
            small_coords.append((np.array([[[x1, y1]], [[x2, y1]], [[x2, y2]], [[x1, y2]]], np.int32), height * width))
        else:
            medium_large_coords.append((np.array([[[x1, y1]], [[x2, y1]], [[x2, y2]], [[x1, y2]]], np.int32), height * width))

    return small_coords, medium_large_coords

# ================== DeepLab Utilities / River Segmentation ==================
CLASSES = {0: 'non-river', 1: 'river'}

"""
Loads the DeepLab model using the fine-tuned weights
"""
def load_deeplab(device):
    model = MSC(
        base=DeepLabV2(
            n_classes=len(CLASSES), n_blocks=[3, 4, 23, 3], atrous_rates=[6, 12, 18, 24]
        ),
        scales=[0.5, 0.75],
    )

    state_dict = torch.load("./model/deeplab.pth", map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict)
    model = torch.nn.DataParallel(model)
    model.eval()
    model.to(device)

    return model

"""
Infers the DeepLab v2 model
"""
def infer_deeplab(image, raw_image, device):
    model = load_deeplab(device)

    H, W, _ = raw_image.shape

    # Image -> Probability map
    logits = model(image)
    logits = F.interpolate(logits, size=(
        H, W), mode="bilinear", align_corners=False)
    probs = F.softmax(logits, dim=1)[0]
    probs = probs.detach().cpu().numpy()

    labelmap = np.argmax(probs, axis=0)
    labelmap = (~labelmap.astype(bool)).astype(int)

    total_river_pixels = (H * W) - cv2.countNonZero(labelmap)
    # check if image is a river (river pixel > 40% of the image)
    # if (total_river_pixels < (H * W) * 0.40):
    #     return [], []

    w_mask = white_mask(labelmap)
    raw_image = cv2.addWeighted(raw_image, 1, w_mask, 1, 0)
    
    labelmap = ndimage.binary_fill_holes(labelmap - 1)

    return raw_image, labelmap.astype('uint8')