from model import SSD, ResNet
from transform import SSDTransformer
from utils import generate_dboxes, Encoder, colors, coco_classes
from PIL import Image
import logging
import numpy as np
import cv2
import easyocr
import torch
from io import BytesIO
import base64

from debug_utils import draw_detections

logging.basicConfig()
logger = logging.getLogger('log')
logger.setLevel(logging.DEBUG)

args = {
    "nms_threshold": 0.5,
    "cls_threshold": 0.5
}

class StandAloneInference:
    
    def __init__(self, model_path="/home/SSD.pth", download_enabled=True):
        self.model = SSD(backbone=ResNet())

        if torch.cuda.is_available():
                checkpoint = torch.load(model_path)
        else:
                map_location=torch.device('cpu')
                checkpoint = torch.load(model_path, map_location=map_location)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        if torch.cuda.is_available():
            self.model.cuda()
        self.model.eval()
        dboxes = generate_dboxes()
        self.transformer = SSDTransformer(dboxes, (300, 300), val=True)
        self.encoder = Encoder(dboxes)

        self.reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available(), download_enabled=download_enabled)

    def detect(self, img, width, height):
        with torch.no_grad():
            ploc, plabel = self.model(img.unsqueeze(dim=0))
            result = self.encoder.decode_batch(ploc, plabel, args['nms_threshold'], 20)[0]
            loc, label, prob = [r.cpu().numpy() for r in result]
            best = np.argwhere(prob > args['cls_threshold']).squeeze(axis=1)
            loc = loc[best]
            label = label[best]
            prob = prob[best]

            logger.debug(label)
            logger.debug(loc)
            logger.debug(prob)

            originalLocations = []
            for l in loc:
                    l[0] *= width
                    l[2] *= width
                    l[1] *= height
                    l[3] *= height

                    originalLocations.append(l.astype(int).tolist())

            detections = {
                    'boxes': originalLocations,
                    'scores': list(prob.tolist()),
                    'classes': list(label.tolist())
            }

            logger.debug(detections)

        return detections


    def ocr(self, img, detections):
        x, y, xmax, ymax = detections['boxes'][0]
        w = xmax - x
        h = ymax - y

        plate_img = img[y:y+h, x:x+w]

        plate_img = cv2.cvtColor(plate_img, cv2.COLOR_RGB2HSV)

        # equalize the histogram of the Y channel
        plate_img[:,:,2] = cv2.equalizeHist(plate_img[:,:,2])

        # convert the YUV image back to RGB format
        plate_img = cv2.cvtColor(plate_img, cv2.COLOR_HSV2RGB)

        logger.debug(str(w) + ' ' + str(h))
        if w < 100:
            plate_img = cv2.resize(plate_img, (w*3, h*3))
        if w < 200:
            plate_img = cv2.resize(plate_img, (w*2, h*2))

        # Gaussian blur
        plate_img = cv2.GaussianBlur(plate_img, (5, 5), 1)

        plate_img = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)

        cv2.imwrite("plate.jpg", plate_img)

        ocr_raw = self.reader.readtext(plate_img, detail = 0)
        logger.debug(ocr_raw)

        if len(ocr_raw) < 0:
            return ""

        return ocr_raw[0]


    def process_from_disk(self, diskpath):
        with open(diskpath, mode='rb') as f:
            data = f.read()
            
        logger.debug("Loaded image [{}]".format(diskpath))

        return self.process(data)

    def process(self, data):
        if isinstance(data, str):
            data = base64.b64decode(data)
        
        img = Image.open(BytesIO(data)).convert("RGB")

        width = img.width
        height = img.height

        img, _, _, _ = self.transformer(img, None, torch.zeros(1,4), torch.zeros(1))
        if torch.cuda.is_available():
            img = img.cuda()

        # detect plates
        detections = self.detect(img, width, height)

        if len(detections['boxes']) == 0:
            return ""

        # draw_detections(diskpath, detections)

        img_array = np.asarray(bytearray(data), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        ocr_raw = self.ocr(img, detections)
        
        # TODO filter chars
        
        return ocr_raw
