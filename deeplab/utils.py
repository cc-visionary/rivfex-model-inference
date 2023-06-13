import numpy as np
import cv2
import torch

def white_mask(image):
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    r[image == 1], g[image == 1], b[image == 1] = [255, 255, 255]
    mask = np.stack([r, g, b], axis=2)

    return mask

def preprocessing(image):
    # Resize
    image = cv2.resize(np.array(image), dsize=(640, 360), interpolation=cv2.INTER_LINEAR)
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