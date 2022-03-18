import cv2
import numpy as np
import torch
from torchvision import transforms

from . import FaceAlignmentTracker, FaceAlignmentTrackerInput
from ..networks.synergy_net import SynergyNet, ToTensor, Normalize, predict_pose, crop_img


class SynergyNetFaceAlignmentTracker(FaceAlignmentTracker):
    def __init__(self, config):
        super().__init__(config)
        # preprocessing
        self.IMG_SIZE = 120
        self.image_transform = transforms.Compose([ToTensor(), Normalize(mean=127.5, std=128)])

        # model
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.synergy_net = SynergyNet()
        self.synergy_net.to(self.device)
        self.synergy_net.eval()

    def track_face_alignment(self, input_data: FaceAlignmentTrackerInput):
        super(SynergyNetFaceAlignmentTracker, self).track_face_alignment(input_data)
        image = self.input.image
        face_bounding_box = self.input.face_bounding_box

        x, y, width, height = face_bounding_box['x'], face_bounding_box['y'], \
                              face_bounding_box['width'], face_bounding_box['height']

        # enlarge the bbox a little and do a square crop
        roi_box = [x, y, width, height, 0.99]
        w_center = x + width / 2
        h_center = y + height / 2
        side_len = width if width > height else height
        margin = side_len * 1.2 // 2
        roi_box[0], roi_box[1], roi_box[2], roi_box[3] = w_center - margin, h_center - margin, \
                                                         w_center + margin, h_center + margin

        image = crop_img(image, roi_box)
        image = cv2.resize(image, dsize=(self.IMG_SIZE, self.IMG_SIZE), interpolation=cv2.INTER_LINEAR)

        image = self.image_transform(image).unsqueeze(0)
        with torch.no_grad():
            image = image.to(self.device)
            param = self.synergy_net.forward_test(image)
            param = param.squeeze().cpu().numpy().flatten().astype(np.float32)

        angles, translation = predict_pose(param, roi_box)

        return {'yaw': angles[0], 'pitch': angles[1], 'roll': angles[2]}
