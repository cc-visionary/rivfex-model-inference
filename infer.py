from PIL import Image
from scipy import ndimage
from skimage.measure import label, regionprops
from flask_cors import CORS, cross_origin
from dotenv import load_dotenv
from torchvision.ops import box_iou

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import flask
import io
import boto3
import os

load_dotenv()

application = flask.Flask(__name__)
CORS(application)

model = None
CLASSES = {0: 'non-river', 1: 'river'}
DEVICE = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')
MODEL_PATH = "./model/fine_tune_whole.pth"

source_points = np.array([[180, 100], [460, 100], [120, 260], [520, 260]], dtype=np.float32)
destination_points = np.array([[220, 150], [420, 150], [220, 310], [420, 310]], dtype=np.float32)
M = cv2.getPerspectiveTransform(source_points, destination_points)

def initalize():
    global model
    model = MSC(
        base=DeepLabV2(
            n_classes=len(CLASSES), n_blocks=[3, 4, 23, 3], atrous_rates=[6, 12, 18, 24]
        ),
        scales=[0.5, 0.75],
    )

    state_dict = torch.load(
        MODEL_PATH, map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict)
    model = torch.nn.DataParallel(model)
    model.eval()
    model.to(DEVICE)


def white_mask(image):
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    r[image == 1], g[image == 1], b[image == 1] = [255, 255, 255]
    mask = np.stack([r, g, b], axis=2)

    return mask


def preprocessing(image):
    # Resize
    image = cv2.resize(image, dsize=(
        640, 360), interpolation=cv2.INTER_LINEAR)
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


_BATCH_NORM = nn.BatchNorm2d
_BOTTLENECK_EXPANSION = 4


class _ConvBnReLU(nn.Sequential):
    """
    Cascade of 2D convolution, batch norm, and ReLU.
    """
    BATCH_NORM = _BATCH_NORM

    def __init__(
        self, in_ch, out_ch, kernel_size, stride, padding, dilation, relu=True
    ):
        super(_ConvBnReLU, self).__init__()
        self.add_module(
            "conv",
            nn.Conv2d(
                in_ch, out_ch, kernel_size, stride, padding, dilation, bias=False
            ),
        )
        self.add_module("bn", _BATCH_NORM(
            out_ch, eps=1e-5, momentum=1 - 0.999))

        if relu:
            self.add_module("relu", nn.ReLU())


class _Bottleneck(nn.Module):
    """
    Bottleneck block of MSRA ResNet.
    """

    def __init__(self, in_ch, out_ch, stride, dilation, downsample):
        super(_Bottleneck, self).__init__()
        mid_ch = out_ch // _BOTTLENECK_EXPANSION
        self.reduce = _ConvBnReLU(in_ch, mid_ch, 1, stride, 0, 1, True)
        self.conv3x3 = _ConvBnReLU(
            mid_ch, mid_ch, 3, 1, dilation, dilation, True)
        self.increase = _ConvBnReLU(mid_ch, out_ch, 1, 1, 0, 1, False)
        self.shortcut = (
            _ConvBnReLU(in_ch, out_ch, 1, stride, 0, 1, False)
            if downsample
            else nn.Identity()
        )

    def forward(self, x):
        h = self.reduce(x)
        h = self.conv3x3(h)
        h = self.increase(h)
        h += self.shortcut(x)
        return F.relu(h)


class _ResLayer(nn.Sequential):
    """
    Residual layer with multi grids
    """

    def __init__(self, n_layers, in_ch, out_ch, stride, dilation, multi_grids=None):
        super(_ResLayer, self).__init__()

        if multi_grids is None:
            multi_grids = [1 for _ in range(n_layers)]
        else:
            assert n_layers == len(multi_grids)

        # Downsampling is only in the first block
        for i in range(n_layers):
            self.add_module(
                "block{}".format(i + 1),
                _Bottleneck(
                    in_ch=(in_ch if i == 0 else out_ch),
                    out_ch=out_ch,
                    stride=(stride if i == 0 else 1),
                    dilation=dilation * multi_grids[i],
                    downsample=(True if i == 0 else False),
                ),
            )


class _Stem(nn.Sequential):
    """
    The 1st conv layer.
    Note that the max pooling is different from both MSRA and FAIR ResNet.
    """

    def __init__(self, out_ch):
        super(_Stem, self).__init__()
        self.add_module("conv1", _ConvBnReLU(3, out_ch, 7, 2, 3, 1))
        self.add_module("pool", nn.MaxPool2d(3, 2, 1, ceil_mode=True))


class ResNet(nn.Sequential):
    def __init__(self, n_classes, n_blocks):
        super(ResNet, self).__init__()
        ch = [64 * 2 ** p for p in range(6)]
        self.add_module("layer1", _Stem(ch[0]))
        self.add_module("layer2", _ResLayer(n_blocks[0], ch[0], ch[2], 1, 1))
        self.add_module("layer3", _ResLayer(n_blocks[1], ch[2], ch[3], 2, 1))
        self.add_module("layer4", _ResLayer(n_blocks[2], ch[3], ch[4], 2, 1))
        self.add_module("layer5", _ResLayer(n_blocks[3], ch[4], ch[5], 2, 1))
        self.add_module("pool5", nn.AdaptiveAvgPool2d(1))
        self.add_module("flatten", nn.Flatten())
        self.add_module("fc", nn.Linear(ch[5], n_classes))


class MSC(nn.Module):
    """
    Multi-scale inputs
    """

    def __init__(self, base, scales=None):
        super(MSC, self).__init__()
        self.base = base
        if scales:
            self.scales = scales
        else:
            self.scales = [0.5, 0.75]

    def forward(self, x):
        # Original
        logits = self.base(x)
        _, _, H, W = logits.shape

        def interp(l): return F.interpolate(
            l, size=(H, W), mode="bilinear", align_corners=False
        )

        # Scaled
        logits_pyramid = []
        for p in self.scales:
            h = F.interpolate(x, scale_factor=p,
                              mode="bilinear", align_corners=False)
            logits_pyramid.append(self.base(h))

        # Pixel-wise max
        logits_all = [logits] + [interp(l) for l in logits_pyramid]
        logits_max = torch.max(torch.stack(logits_all), dim=0)[0]

        if self.training:
            return [logits] + logits_pyramid + [logits_max]
        else:
            return logits_max


class _ASPP(nn.Module):
    """
    Atrous spatial pyramid pooling (ASPP)
    """

    def __init__(self, in_ch, out_ch, rates):
        super(_ASPP, self).__init__()
        for i, rate in enumerate(rates):
            self.add_module(
                "c{}".format(i),
                nn.Conv2d(in_ch, out_ch, 3, 1, padding=rate,
                          dilation=rate, bias=True),
            )

        for m in self.children():
            nn.init.normal_(m.weight, mean=0, std=0.01)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return sum([stage(x) for stage in self.children()])


class DeepLabV2(nn.Sequential):
    """
    DeepLab v2: Dilated ResNet + ASPP
    Output stride is fixed at 8
    """

    def __init__(self, n_classes, n_blocks, atrous_rates):
        super(DeepLabV2, self).__init__()
        ch = [64 * 2 ** p for p in range(6)]
        self.add_module("layer1", _Stem(ch[0]))
        self.add_module("layer2", _ResLayer(n_blocks[0], ch[0], ch[2], 1, 1))
        self.add_module("layer3", _ResLayer(n_blocks[1], ch[2], ch[3], 2, 1))
        self.add_module("layer4", _ResLayer(n_blocks[2], ch[3], ch[4], 1, 2))
        self.add_module("layer5", _ResLayer(n_blocks[3], ch[4], ch[5], 1, 4))
        self.add_module("aspp", _ASPP(ch[5], n_classes, atrous_rates))

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, _ConvBnReLU.BATCH_NORM):
                m.eval()


def segment_river(image, raw_image):
    _, _, H, W = image.shape

    # Image -> Probability map
    logits = model(image)
    logits = F.interpolate(logits, size=(
        H, W), mode="bilinear", align_corners=False)
    probs = F.softmax(logits, dim=1)[0]
    probs = probs.detach().cpu().numpy()

    labelmap = np.argmax(probs, axis=0)

    labelmap = (~labelmap.astype(bool)).astype(int)
    labelmap = ndimage.binary_fill_holes(1 - labelmap).astype(int)
    labelmap = ndimage.binary_fill_holes(1 - labelmap).astype(int)
    labelmap = ndimage.binary_dilation(labelmap).astype(int)

    total_river_pixels = (H * W) - cv2.countNonZero(labelmap)
    # check if image is a river (river pixel > 40% of the image)
    if (total_river_pixels < (H * W) * 0.40):
        return None, None

    w_mask = white_mask(labelmap)
    raw_image = cv2.addWeighted(raw_image, 1, w_mask, 1, 0)

    return raw_image, labelmap - 1

# performs image processing techniques
# remove small bounding boxes inside of bigger boxes
# then performs perspective transformation on the image
def threshold_hsv(segmented, raw_img, river_mask):
    hsv_img = cv2.cvtColor(segmented, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv_img, np.array(
        [60, 100, 0]), np.array([90, 150, 160]))

    # Noise removal using Morphological open operation
    kernel = np.ones((3, 3), np.uint8)
    morphOpen = cv2.dilate(mask, kernel, iterations=2)
    kernel = np.ones((3, 3), np.uint8)
    morphOpen = cv2.dilate(morphOpen, kernel, iterations=1)
    morphClose = cv2.morphologyEx(
        morphOpen, cv2.MORPH_CLOSE, kernel, iterations=1)
    morphOpen = cv2.morphologyEx(
        morphClose, cv2.MORPH_OPEN, kernel, iterations=1)

    labels = regionprops(label(morphOpen))

    bboxed_img = raw_img.copy()

    bboxes = []
    for i in labels:
        minr, minc, maxr, maxc = i.bbox
        if (maxr-minr > 5 and maxc-minc > 5):
            bboxes.append([minc, minr, maxc, maxr])

    bboxes = torch.Tensor(bboxes)

    # remove box within boxes
    if (len(bboxes) > 0):
        pairwise = box_iou(bboxes, bboxes).numpy()
        xs, ys = np.where((pairwise == pairwise) & (
            pairwise != 0) & (pairwise != 1))
        to_be_removed = []
        for x, y in zip(xs, ys):
            if ((bboxes[x][2] - bboxes[x][0]) * (bboxes[x][3] - bboxes[x][1]) > (bboxes[y][2] - bboxes[y][0]) * (bboxes[y][3] - bboxes[y][1])):
                if (y not in to_be_removed):
                    to_be_removed.append(y)
            else:
                if (x not in to_be_removed):
                    to_be_removed.append(x)

        indices = list(np.arange(len(bboxes)))
        for index in to_be_removed:
            indices.remove(index)

        bboxes = bboxes[indices]

    # warp river mask
    warped_river = cv2.warpPerspective(river_mask.astype('uint8'), M, (640, 360), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    transformed_river_area = cv2.countNonZero(warped_river)

    # get warped, perspective transorm
    warped = cv2.warpPerspective(raw_img, M, (640, 360), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    total_obstructed_river_pixels = 0
    total_transformed_obstructed_river_pixels = 0
    for minc, minr, maxc, maxr in bboxes:
        total_obstructed_river_pixels += (maxc - minc) * (maxr - minr)
    
        custom_box = np.array([[[minc, minr]], [[maxc, minr]], [[minc, maxr]], [[maxc, maxr]]])
        warped_box = cv2.perspectiveTransform(custom_box, M)
        total_transformed_obstructed_river_pixels += (warped_box[3][0][0] - warped_box[0][0][0]) * (warped_box[3][0][1] - warped_box[0][0][1])

        cv2.rectangle(bboxed_img, (int(minc), int(minr)), (int(maxc), int(maxr)), (255, 0, 0), 1)
        cv2.rectangle(warped, (int(warped_box[0][0][0]), int(warped_box[0][0][1])), (int(warped_box[3][0][0]), int(warped_box[3][0][1])), (255, 0, 0), 1)
        
    # returns bboxed image, warped bbox image, actual river covered, transformed river covered
    return bboxed_img, warped, min(total_obstructed_river_pixels / cv2.countNonZero(river_mask) * 100, 100), min(total_transformed_obstructed_river_pixels / transformed_river_area * 100, 100)


@application.route("/inference/delete", methods=["DELETE"])
@cross_origin()
def delete():
    try:
        if flask.request.method == "DELETE":
            s3_client = boto3.client("s3", aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                                     aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"), region_name=os.getenv("REGION"))

            try:
                for folder in ['original/', 'segmented/', 'bboxed/', 'transformed/']:
                    s3_client.delete_object(
                        Bucket=os.getenv("BUCKET"), Key=folder + flask.request.args['filename'])
            except Exception as e:
                return flask.jsonify({"success": False, "message": repr(e)})

            return flask.jsonify({"success": True, "message": "Deleted Images from AWS S3 Successfully"})
    except Exception as e:
        return flask.jsonify({"success": False, "message": repr(e)})


@application.route("/inference/predict", methods=["POST"])
@cross_origin()
def predict():
    if flask.request.method == "POST":
        try:
            img = np.array(Image.open(flask.request.files['image']))
            image, raw_image = preprocessing(img)

            segmented, river_mask = segment_river(image, raw_image)

            if (type(segmented) == np.ndarray):
                # Upload the file
                s3_client = boto3.resource("s3", aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                                           aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"), region_name=os.getenv("REGION")).Bucket(os.getenv("BUCKET"))

                # image processing for surface object detection
                bboxed_img, warped_img, actual_river_covered, transformed_river_covered = threshold_hsv(
                    segmented, raw_image, river_mask)

                # upload images to AWS EC2
                for folder, img in [('original', raw_image), ('segmented', segmented), ('bboxed', bboxed_img), ('transformed', warped_img)]:
                    try:
                        # convert numpy array to PIL Image
                        im = Image.fromarray(img.astype('uint8'))

                        # create file-object in memory
                        file = io.BytesIO()

                        # write PNG in file-object
                        im.save(file, 'jpeg')

                        # move to beginning of file so `send_file()` it will read from start
                        file.seek(0)

                        s3_client.put_object(
                            Key=("%s/" % (folder)) + flask.request.files['image'].filename, Body=file, ContentType='image/jpeg')
                    except Exception as e:
                        return flask.jsonify({"success": False, "message": repr(e)})
                return flask.jsonify({
                    "success": True,
                    "message": "Segmented and Uploaded Successfully",
                    "actual_percentage_river_covered": float(actual_river_covered),
                    "transformed_percentage_river_covered": float(transformed_river_covered),
                })
            return flask.jsonify({"success": False, "message": "Detected segmented river area is less than 50% of the image...<br />Possible Causes:<br />1. Uploaded image might not be a river image<br />2. River is covered by something else.<br />Please try again..."})
        except Exception as e:
            return flask.jsonify({"success": False, "message": repr(e)})

    return False


if __name__ == "__main__":
    print("Loading inference server")
    print("Utilizing %s for computing" % DEVICE)
    initalize()
    application.run(host='localhost', port='8000')
