from typing import List

import dlib
import face_alignment
import face_alignment.detection.sfd
from facenet_pytorch import MTCNN
import numpy as np
import yacs.config

from ptgaze import Face



class LandmarkEstimator:
    def __init__(self, config: yacs.config.CfgNode):
        self.mode = config.face_detector.mode
        if self.mode == 'dlib':
            self.detector = dlib.get_frontal_face_detector()
            self.predictor = dlib.shape_predictor(
                config.face_detector.dlib.model)

        if self.mode == 'mtcnn':
              self.detector = MTCNN(select_largest=False, post_process=False)
              self.predictor = dlib.shape_predictor(
                   config.face_detector.dlib.model)

        elif self.mode == 'face_alignment_dlib':
            self.detector = dlib.get_frontal_face_detector()
            self.predictor = face_alignment.FaceAlignment(
                face_alignment.LandmarksType._2D,
                face_detector='dlib',
                flip_input=False,
                device=config.device)
        elif self.mode == 'face_alignment_sfd':
            self.detector = face_alignment.detection.sfd.sfd_detector.SFDDetector(
                device=config.device)
            self.predictor = face_alignment.FaceAlignment(
                face_alignment.LandmarksType._2D,
                flip_input=False,
                device=config.device)

        else:
            raise ValueError

    def detect_faces(self, image: np.ndarray) -> List[Face]:
        if self.mode == 'dlib':
            return self._detect_faces_dlib(image)
        elif self.mode == 'face_alignment_dlib':
            return self._detect_faces_face_alignment_dlib(image)
        elif self.mode == 'face_alignment_sfd':
            return self._detect_faces_face_alignment_sfd(image)
        elif self.mode == 'mtcnn':
            return self._detect_faces_mtcnn(image)
        else:
            raise ValueError

    def _detect_faces_dlib(self, image: np.ndarray) -> List[Face]:
        bboxes = self.detector(image[:, :, ::-1], 0)
        detected = []
        for bbox in bboxes:
            predictions = self.predictor(image[:, :, ::-1], bbox)
            landmarks = np.array([(pt.x, pt.y) for pt in predictions.parts()],
                                 dtype=np.float)
            bbox = np.array([[bbox.left(), bbox.top()],
                             [bbox.right(), bbox.bottom()]],
                            dtype=np.float)
            detected.append(Face(bbox, landmarks))
        return detected

    def _detect_faces_mtcnn(self, image: np.ndarray) -> List[Face]:# MTCNN
        bboxes, _ = self.detector.detect(image[:, :, ::-1])
        detected = []
        for bbox in bboxes:
            det = dlib.rectangle(int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
            predictions = self.predictor(image[:, :, ::-1], det)
            landmarks = np.array([(pt.x, pt.y) for pt in predictions.parts()],
                                 dtype=np.float)
            bbox = np.array([[bbox[0], bbox[1]],
                             [bbox[2], bbox[3]]],
                            dtype=np.float)
            detected.append(Face(bbox, landmarks))
        return detected

    def _detect_faces_face_alignment_dlib(self,
                                          image: np.ndarray) -> List[Face]:
        bboxes = self.detector(image[:, :, ::-1], 0)
        bboxes = [[bbox.left(),
                   bbox.top(),
                   bbox.right(),
                   bbox.bottom()] for bbox in bboxes]
        predictions = self.predictor.get_landmarks(image[:, :, ::-1],
                                                   detected_faces=bboxes)
        if predictions is None:
            predictions = []
        detected = []
        for bbox, landmarks in zip(bboxes, predictions):
            bbox = np.array(bbox, dtype=np.float).reshape(2, 2)
            detected.append(Face(bbox, landmarks))
        return detected

    def _detect_faces_face_alignment_sfd(self,
                                         image: np.ndarray) -> List[Face]:
        bboxes = self.detector.detect_from_image(image[:, :, ::-1].copy())
        bboxes = [bbox[:4] for bbox in bboxes]
        predictions = self.predictor.get_landmarks(image[:, :, ::-1],
                                                   detected_faces=bboxes)
        if predictions is None:
            predictions = []
        detected = []
        for bbox, landmarks in zip(bboxes, predictions):
            bbox = np.array(bbox, dtype=np.float).reshape(2, 2)
            detected.append(Face(bbox, landmarks))
        return detected
