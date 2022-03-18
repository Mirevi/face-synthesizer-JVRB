from .face_aabbox_tracking import FaceAABBoxTracker, FaceAABBoxTrackerInput
from ..networks.synergy_net import FaceBoxes


class SynergyNetFaceAABBoxTracker(FaceAABBoxTracker):
    def __init__(self, config):
        super().__init__(config)
        self.face_box_predictor = FaceBoxes()

    def track_face_aabbox(self, input_data: FaceAABBoxTrackerInput):
        super(SynergyNetFaceAABBoxTracker, self).track_face_aabbox(input_data)
        image = self.input_data.image

        # track face boxes
        faces_bounding_boxes = self.face_box_predictor(image)

        # if len not 1 then return None
        if len(faces_bounding_boxes) != 1:
            return None  # No face or too many faces

        bounding_box = faces_bounding_boxes[0]

        return {
            'x': bounding_box[0],
            'y': bounding_box[1],
            'width': bounding_box[2] - bounding_box[0],
            'height': bounding_box[3] - bounding_box[1],
        }


