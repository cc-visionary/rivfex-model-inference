from deeplab.deeplabv2 import DeepLabV2
from deeplab.msc import MSC
from deeplab.utils import white_mask, preprocessing

from PIL import Image
from scipy import ndimage
from skimage.measure import label, regionprops
from flask import send_file
from flask_cors import CORS, cross_origin

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import numpy as np
import flask
import os
import io
import urllib
import boto3

app = flask.Flask(__name__)
CORS(app)

model = None
# MODEL_DIR = os.getenv("MODEL_DIR")
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
            s3_client = boto3.client("s3", aws_access_key_id="AKIAWIZJYQWT2GDAX74U",
                                       aws_secret_access_key="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")
            
            try:
                s3_client.delete_object(Bucket="rivfex-original", Key=flask.request.args['filename'])
                s3_client.delete_object(Bucket="rivfex-segmented", Key=flask.request.args['filename'])
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
                s3_client = boto3.resource("s3", aws_access_key_id="AKIAWIZJYQWT2GDAX74U",
                                           aws_secret_access_key="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")

                try:
                    # convert numpy array to PIL Image
                    result_img = Image.fromarray(result.astype('uint8'))

                    # create file-object in memory
                    result_file = io.BytesIO()

                    # write PNG in file-object
                    result_img.save(result_file, 'jpeg')

                    # move to beginning of file so `send_file()` it will read from start
                    result_file.seek(0)

                    s3_client.Bucket("rivfex-segmented").put_object(
                        Key=flask.request.files['image'].filename, Body=result_file, ContentType='image/jpeg')
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

                    s3_client.Bucket("rivfex-original").put_object(
                        Key=flask.request.files['image'].filename, Body=result_file, ContentType='image/jpeg')
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
    app.run()
