from PIL import Image
from scipy import ndimage
from skimage.measure import label, regionprops
from flask_cors import CORS, cross_origin
from dotenv import load_dotenv

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

app = flask.Flask(__name__)
CORS(app)

model = None
CLASSES = {0: 'non-river', 1: 'river'}
DEVICE = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')
MODEL_PATH = "./model/3_fold_normalized.pth"


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
    image = cv2.resize(np.array(image), dsize=(
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


def segment_river(img):
    image, raw_image = preprocessing(img)

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

    # check if image is a rivel
    if (cv2.countNonZero(labelmap) > (H * W) / 2):
        return None

    w_mask = white_mask(labelmap)
    raw_image = cv2.addWeighted(raw_image, 1, w_mask, 1, 0)

    return raw_image


def threshold_hsv(raw_img):
    hsv_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv_img, np.array(
        [50, 60, 65]), np.array([85, 120, 165]))

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

    bboxes = []
    for i in labels:
        minr, minc, maxr, maxc = i.bbox
        if (maxr-minr > 5 and maxc-minc > 5):
            bboxes.append([minc, minr, maxc, maxr])

    return bboxes


@app.route("/delete", methods=["DELETE"])
@cross_origin()
def delete():
    try:
        if flask.request.method == "DELETE":
            s3_client = boto3.client("s3", aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                                     aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"))

            try:
                s3_client.delete_object(
                    Bucket=os.getenv("BUCKET"), Key="original/" + flask.request.args['filename'])
                s3_client.delete_object(
                    Bucket=os.getenv("BUCKET"), Key="segmented/" + flask.request.args['filename'])
            except Exception as e:
                return flask.jsonify({"success": False, "message": repr(e)})

            return flask.jsonify({"success": True, "message": "Deleted Images from AWS S3 Successfully"})
    except Exception as e:
        return flask.jsonify({"success": False, "message": repr(e)})


@app.route("/infer", methods=["POST"])
@cross_origin()
def predict():
    if flask.request.method == "POST":
        try:
            image = np.array(Image.open(flask.request.files['image']))

            result = segment_river(image)

            if (type(result) == np.ndarray):
                # Upload the file
                s3_client = boto3.resource("s3", aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                                           aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"))

                try:
                    # convert numpy array to PIL Image
                    result_img = Image.fromarray(result.astype('uint8'))

                    # create file-object in memory
                    result_file = io.BytesIO()

                    # write PNG in file-object
                    result_img.save(result_file, 'jpeg')

                    # move to beginning of file so `send_file()` it will read from start
                    result_file.seek(0)

                    s3_client.Bucket(os.getenv("BUCKET")).put_object(
                        Key="segmented/" + flask.request.files['image'].filename, Body=result_file, ContentType='image/jpeg')
                except Exception as e:
                    return flask.jsonify({"success": False, "message": repr(e)})

                try:
                    # convert numpy array to PIL Image
                    result_img = Image.fromarray(image.astype('uint8'))

                    # create file-object in memory
                    result_file = io.BytesIO()

                    # write PNG in file-object
                    result_img.save(result_file, 'jpeg')

                    # move to beginning of file so `send_file()` it will read from start
                    result_file.seek(0)

                    s3_client.Bucket(os.getenv("BUCKET")).put_object(
                        Key="original/" + flask.request.files['image'].filename, Body=result_file, ContentType='image/jpeg')
                except Exception as e:
                    return flask.jsonify({"success": False, "message": repr(e)})

                # image processing for surface object detection
                bboxes = threshold_hsv(result)

                return flask.jsonify({"success": True, "message": "Segmented and Uploaded Successfully", "bboxes": bboxes})
            return flask.jsonify({"success": False, "message": "River Mask is less than 50%% of the image"})
        except Exception as e:
            return flask.jsonify({"success": False, "message": repr(e)})

    return False


if __name__ == "__main__":
    print("Loading inference server")
    print("Utilizing %s for computing" % DEVICE)
    initalize()
    app.run(host='0.0.0.0', port='3000')
