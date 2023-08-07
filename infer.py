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

from utilities import infer_deeplab, infer_yolov7, infer_thresholding, create_overlay, preprocessing

load_dotenv()

application = flask.Flask(__name__)
CORS(application)

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

source_points = np.array([[210, 100], [430, 100], [120, 260], [520, 260]], dtype=np.float32)
destination_points = np.array([[520, 400], [720, 400], [520, 560], [720, 560]], dtype=np.float32)
M = cv2.getPerspectiveTransform(source_points, destination_points)

"""
Infers the Whole System
1. performs river segmentation
2. performs water hyacinth detection along with garbage detection,
3. performs perspective transformation on the image
4. calculates the percentage of river covere by river surface objects
5. returns segmented, annotated, and warped images, along with percentage of river covered
"""
def infer_system(img, raw_img):
    height, width, _ = raw_img.shape
    segmented_img, river_mask = infer_deeplab(img, raw_img, DEVICE)
    if(type(segmented_img) == np.ndarray):
        g_small_coords, g_med_large_coords = infer_yolov7(segmented_img, DEVICE)
        wh_small_coords, wh_med_large_coords = infer_thresholding(segmented_img)

        predicted_obj_mask = np.zeros((height, width))
        detected_img = raw_img.copy()

        g_count = len(g_med_large_coords)
        wh_count = len(wh_med_large_coords)

        print(g_count, wh_count)

        # if small_objects collectively > 32 * 32
        total_small_area = sum([area for _, area in [*g_small_coords, *wh_small_coords]])
        if(total_small_area > 32 * 32):
            g_count += len(g_small_coords)
            wh_count += len(wh_small_coords)
            for small_coords in [*g_small_coords, *wh_small_coords]:
                coord, _ = small_coords
                cv2.drawContours(predicted_obj_mask, [coord], 0, 1, -1)

        for med_large_coords in [*g_med_large_coords, *wh_med_large_coords]:
            coord, _ = med_large_coords
            cv2.drawContours(predicted_obj_mask, [coord], 0, 1, -1)

        total_obj_area = cv2.countNonZero(predicted_obj_mask)

        # warp detected 
        warped_obj_mask = cv2.warpPerspective(predicted_obj_mask, M, (1240, 610), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        warped_img = cv2.warpPerspective(raw_img, M, (1240, 610), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)    
        total_warped_obj_area = cv2.countNonZero(warped_obj_mask)
            
        # warp river mask
        warped_river = cv2.warpPerspective(river_mask.astype('uint8'), M, (1240, 610), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        transformed_river_area = cv2.countNonZero(warped_river)

        detected_img = cv2.addWeighted(detected_img, 1, create_overlay(predicted_obj_mask), 0.25, 0)
        warped_img = cv2.addWeighted(warped_img, 1, create_overlay(warped_obj_mask), 0.25, 0)
        
        river_covered = min(total_obj_area / cv2.countNonZero(river_mask) * 100, 100)
        transformed_river_covered = min(total_warped_obj_area / transformed_river_area * 100, 100)

        # returns bboxed image, warped bbox image, actual river covered, transformed river covered
        return segmented_img, detected_img, warped_img, river_covered, transformed_river_covered
    else:
        return None, None, None, None, None

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
            img = Image.open(flask.request.files['image']).convert("RGB")
            img = np.array(img)
            image, raw_image = preprocessing(img)

            # locate objects while retrieving river covered
            segmented, detected_img, warped_img, actual_river_covered, transformed_river_covered = infer_system(image, raw_image)

            if (type(detected_img) == np.ndarray):
                # Upload the file
                s3_client = boto3.resource("s3", aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                                           aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"), region_name=os.getenv("REGION")).Bucket(os.getenv("BUCKET"))

                # upload images to AWS EC2
                for folder, img in [('original', raw_image), ('segmented', segmented), ('detected', detected_img), ('transformed', warped_img)]:
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
    application.run(host='localhost', port='8000')
