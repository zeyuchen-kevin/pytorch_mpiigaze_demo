from typing import Optional

import datetime
import logging
import pathlib
import os

import time
import dlib
import cv2
import numpy as np
import yacs.config
from facenet_pytorch import MTCNN

from ptgaze import (Face, FacePartsName, GazeEstimationMethod, GazeEstimator,
                    Visualizer)

from ptgaze.utils import intersec_boxes


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Demo:
    QUIT_KEYS = {27, ord('q')}

    def __init__(self, config: yacs.config.CfgNode):
        self.config = config
        self.gaze_estimator = GazeEstimator(config)
        self.visualizer = Visualizer(self.gaze_estimator.camera)

        self.cap = self._create_capture()
        self.output_dir = self._create_output_dir()
        self.writer = self._create_video_writer()

        self.stop = False
        self.show_bbox = self.config.demo.show_bbox
        self.show_head_pose = self.config.demo.show_head_pose
        self.show_landmarks = self.config.demo.show_landmarks
        self.show_normalized_image = self.config.demo.show_normalized_image
        self.show_template_model = self.config.demo.show_template_model


    def run(self) -> None:
        # self._log_out()
        if self.config.demo.use_camera or self.config.demo.video_path:
            self._run_on_video()
        elif self.config.demo.image_path:
            if pathlib.Path(self.config.demo.image_path).is_dir():
                self._run_on_imagedir()
            else:
                self._run_on_image()
        else:
            raise ValueError

    def _run_on_image(self):
        image = cv2.imread(self.config.demo.image_path)
        self._process_image(image)
        if self.config.demo.display_on_screen:
            while True:
                key_pressed = self._wait_key()
                if self.stop:
                    break
                if key_pressed:
                    self._process_image(image)
                cv2.imshow('image', self.visualizer.image)
        if self.config.demo.output_dir:
            name = pathlib.Path(self.config.demo.image_path).name
            output_path = pathlib.Path(self.config.demo.output_dir) / name
            cv2.imwrite(output_path.as_posix(), self.visualizer.image)

    # def _run_on_imagedir(self):
    #     dir_name = str(self.config.demo.image_path).split('/')[-1]
    #     img_list = os.listdir(str(self.config.demo.image_path))
    #     img_list.sort(key=lambda x: int(x.split('.')[0]))  # sort files by ascending
    #
    #
    #     detector = dlib.get_frontal_face_detector()
    #     predictor = dlib.shape_predictor(self.config.face_detector.dlib.model)
    #     tracker = dlib.correlation_tracker()
    #     tracking_state = False
    #
    #     for img_name in img_list:
    #         image_path = os.path.join(str(self.config.demo.image_path), img_name)
    #         image = cv2.imread(image_path)
    #         undistorted = cv2.undistort(
    #             image, self.gaze_estimator.camera.camera_matrix,
    #             self.gaze_estimator.camera.dist_coefficients)
    #         self.visualizer.set_image(image.copy())
    #
    #         bboxes = detector(undistorted[:, :, ::-1], 0)
    #         faces = []
    #         if tracking_state is False:
    #             if len(bboxes) > 0:
    #                 for bbox in bboxes:
    #                     tracking_state = True
    #                     tracker.start_track(undistorted[:, :, ::-1], bbox)
    #                     predictions = predictor(undistorted[:, :, ::-1], bbox)
    #                     landmarks = np.array([(pt.x, pt.y) for pt in predictions.parts()],
    #                                          dtype=np.float)
    #                     bbox = np.array([[bbox.left(), bbox.top()],
    #                                      [bbox.right(), bbox.bottom()]],
    #                                     dtype=np.float)
    #                     faces.append(Face(bbox, landmarks))
    #         else:
    #             tracker.update(undistorted[:, :, ::-1])
    #             pos = tracker.get_position()
    #
    #
    #             if len(bboxes) > 0:
    #                 max_inter = - 1
    #                 max_inter_id = -1
    #                 for i, box in enumerate(bboxes):
    #                     inter = intersec_boxes(box, pos)
    #                     if inter != 0 and inter > max_inter:
    #                         max_inter = inter
    #                         max_inter_id = i
    #
    #                 if max_inter_id != -1:
    #                     predictions = predictor(undistorted[:, :, ::-1], bboxes[max_inter_id])
    #                     landmarks = np.array([(pt.x, pt.y) for pt in predictions.parts()],
    #                                          dtype=np.float)
    #                     bbox = np.array([[bboxes[max_inter_id].left(), bboxes[max_inter_id].top()],
    #                                      [bboxes[max_inter_id].right(), bboxes[max_inter_id].bottom()]],
    #                                     dtype=np.float)
    #                     faces.append(Face(bbox, landmarks))
    #
    #         for face_id, face in enumerate(faces):
    #             self.gaze_estimator.estimate_gaze(undistorted, face)
    #             self._draw_face_bbox(face)
    #             self._draw_head_pose(face)
    #             self._draw_landmarks(face)
    #             self._draw_face_template_model(face)
    #             self._draw_gaze_vector(face)
    #             self._display_normalized_image(face)
    #
    #             gaze_vector = []
    #             for key in [FacePartsName.REYE, FacePartsName.LEYE]:
    #                 eye = getattr(face, key.name.lower())
    #                 gaze_vector.append(eye.gaze_vector)
    #             with open('./result/' + dir_name + '.txt', 'a+') as f:
    #                 f.write(str(img_name) + ' ' + str(face_id) + ' ' +
    #                         str(gaze_vector[0][0]) + ' ' + str(gaze_vector[0][1]) + ' ' + str(gaze_vector[0][2]) + ' ' +
    #                         str(gaze_vector[1][0]) + ' ' + str(gaze_vector[1][1]) + ' ' + str(gaze_vector[1][2]))
    #                 f.write("\n")
    #
    #         if self.writer:
    #             self.writer.write(self.visualizer.image)
                
    def _run_on_imagedir(self):
        dir_name = str(self.config.demo.image_path).split('/')[-1]
        img_list = os.listdir(str(self.config.demo.image_path))
        img_list.sort(key=lambda x: int(x.split('.')[0]))  # sort files by ascending


        detector = MTCNN(select_largest=False, post_process=False)
        predictor = dlib.shape_predictor(self.config.face_detector.dlib.model)
        tracking_state = False

        for img_name in img_list:
            image_path = os.path.join(str(self.config.demo.image_path), img_name)
            image = cv2.imread(image_path)
            undistorted = cv2.undistort(
                image, self.gaze_estimator.camera.camera_matrix,
                self.gaze_estimator.camera.dist_coefficients)
            self.visualizer.set_image(image.copy())

            bboxes, _ = detector.detect(undistorted[:, :, ::-1])
            faces = []
            if tracking_state is False:
                if len(bboxes) > 0:
                    for bbox in bboxes:
                        tracking_state = True
                        last_box_center = [np.mean([bbox[0], bbox[2]]), np.mean([bbox[1], bbox[3]])]
                        det = dlib.rectangle(int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
                        predictions = predictor(undistorted[:, :, ::-1], det)
                        landmarks = np.array([(pt.x, pt.y) for pt in predictions.parts()],
                                             dtype=np.float)
                        bbox = np.array([[bbox[0], bbox[1]],
                                         [bbox[2], bbox[3]]],
                                        dtype=np.float)
                        faces.append(Face(bbox, landmarks))
            else:
                if len(bboxes) > 0:
                    min_center = 5000
                    min_center_id = -1
                    for i, bbox in enumerate(bboxes):
                        box_center = [np.mean([bbox[0], bbox[2]]), np.mean([bbox[1], bbox[3]])]
                        box_center_dist = np.linalg.norm(np.array(box_center) - np.array(last_box_center)) ** 2
                        if box_center_dist < min_center:
                            min_center = box_center_dist
                            min_center_id = i

                    if min_center_id != -1:
                        last_box = bboxes[min_center_id]
                        last_box_center = [np.mean([last_box[0], last_box[2]]), np.mean([last_box[1], last_box[3]])]
                        det = dlib.rectangle(int(last_box[0]), int(last_box[1]), int(last_box[2]), int(last_box[3]))
                        predictions = predictor(undistorted[:, :, ::-1], det)
                        landmarks = np.array([(pt.x, pt.y) for pt in predictions.parts()],
                                             dtype=np.float)
                        bbox = np.array([[last_box[0], last_box[1]],
                                         [last_box[2], last_box[3]]],
                                        dtype=np.float)
                        faces.append(Face(bbox, landmarks))

            for face_id, face in enumerate(faces):
                self.gaze_estimator.estimate_gaze(undistorted, face)
                self._draw_face_bbox(face)
                self._draw_head_pose(face)
                self._draw_landmarks(face)
                self._draw_face_template_model(face)
                self._draw_gaze_vector(face)
                self._display_normalized_image(face)

                gaze_vector = []
                for key in [FacePartsName.REYE, FacePartsName.LEYE]:
                    eye = getattr(face, key.name.lower())
                    gaze_vector.append(eye.gaze_vector)
                with open(os.path.join(self.config.demo.output_dir, dir_name + '.txt'), 'a+') as f:
                    f.write(str(img_name) + ' ' + str(face_id) + ' ' +
                            str(gaze_vector[0][0]) + ' ' + str(gaze_vector[0][1]) + ' ' + str(gaze_vector[0][2]) + ' ' +
                            str(gaze_vector[1][0]) + ' ' + str(gaze_vector[1][1]) + ' ' + str(gaze_vector[1][2]))
                    f.write("\n")

            if self.writer:
                self.writer.write(self.visualizer.image)

    def _run_on_video(self) -> None:
        while True:
            if self.config.demo.display_on_screen:
                self._wait_key()
                if self.stop:
                    break

            ok, frame = self.cap.read()
            if not ok:
                break
            self._process_image(frame)

            if self.config.demo.display_on_screen:
                cv2.imshow('frame', self.visualizer.image)
        self.cap.release()
        if self.writer:
            self.writer.release()

    def _process_image(self, image) -> None:
        undistorted = cv2.undistort(
            image, self.gaze_estimator.camera.camera_matrix,
            self.gaze_estimator.camera.dist_coefficients)

        self.visualizer.set_image(image.copy())
        faces = self.gaze_estimator.detect_faces(undistorted)
        for face in faces:
            self.gaze_estimator.estimate_gaze(undistorted, face)
            self._draw_face_bbox(face)
            # self._draw_head_pose(face)
            self._draw_landmarks(face)
            self._draw_face_template_model(face)
            self._draw_gaze_vector(face)
            self._display_normalized_image(face)

        if self.config.demo.use_camera:
            self.visualizer.image = self.visualizer.image[:, ::-1]
        if self.writer:
            self.writer.write(self.visualizer.image)

    def _create_capture(self) -> Optional[cv2.VideoCapture]:
        if self.config.demo.image_path:
            return None
        if self.config.demo.use_camera:
            cap = cv2.VideoCapture(0)
        elif self.config.demo.video_path:
            cap = cv2.VideoCapture(self.config.demo.video_path)
        else:
            raise ValueError
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.gaze_estimator.camera.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.gaze_estimator.camera.height)
        return cap

    def _create_output_dir(self) -> Optional[pathlib.Path]:
        if not self.config.demo.output_dir:
            return
        output_dir = pathlib.Path(self.config.demo.output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        return output_dir

    @staticmethod
    def _create_timestamp() -> str:
        dt = datetime.datetime.now()
        return dt.strftime('%Y%m%d_%H%M%S')

    def _create_video_writer(self) -> Optional[cv2.VideoWriter]:
        if self.config.demo.image_path:
            if pathlib.Path(self.config.demo.image_path).is_dir() is False:
                return None
        if not self.output_dir:
            return None
        ext = self.config.demo.output_file_extension
        if ext == 'mp4':
            # fourcc = cv2.VideoWriter_fourcc(*'H264')
            fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        elif ext == 'avi':
            fourcc = cv2.VideoWriter_fourcc(*'PIM1')
        else:
            raise ValueError
        if self.config.demo.use_camera:
            output_name = f'{self._create_timestamp()}.{ext}'
        elif self.config.demo.video_path:
            name = pathlib.Path(self.config.demo.video_path).stem
            output_name = f'{name}.{ext}'
        elif self.config.demo.image_path:
            name = pathlib.Path(self.config.demo.image_path).stem
            output_name = f'{name}.{ext}'
        else:
            raise ValueError
        output_path = self.output_dir / output_name
        writer = cv2.VideoWriter(output_path.as_posix(), fourcc, 30,
                                 (self.gaze_estimator.camera.width,
                                  self.gaze_estimator.camera.height))
        if writer is None:
            raise RuntimeError
        return writer

    def _wait_key(self) -> bool:
        key = cv2.waitKey(self.config.demo.wait_time) & 0xff
        if key in self.QUIT_KEYS:
            self.stop = True
        elif key == ord('b'):
            self.show_bbox = not self.show_bbox
        elif key == ord('l'):
            self.show_landmarks = not self.show_landmarks
        elif key == ord('h'):
            self.show_head_pose = not self.show_head_pose
        elif key == ord('n'):
            self.show_normalized_image = not self.show_normalized_image
        elif key == ord('t'):
            self.show_template_model = not self.show_template_model
        else:
            return False
        return True

    def _draw_face_bbox(self, face: Face) -> None:
        if not self.show_bbox:
            return
        self.visualizer.draw_bbox(face.bbox)

    def _draw_head_pose(self, face: Face) -> None:
        if not self.show_head_pose:
            return
        # Draw the axes of the model coordinate system
        length = self.config.demo.head_pose_axis_length
        self.visualizer.draw_model_axes(face, length, lw=2)

        euler_angles = face.head_pose_rot.as_euler('XYZ', degrees=True)
        pitch, yaw, roll = face.change_coordinate_system(euler_angles)
        logger.info(f'[head] pitch: {pitch:.2f}, yaw: {yaw:.2f}, '
                    f'roll: {roll:.2f}, distance: {face.distance:.2f}')

    def _draw_landmarks(self, face: Face) -> None:
        if not self.show_landmarks:
            return
        self.visualizer.draw_points(face.landmarks,
                                    color=(0, 255, 255),
                                    size=1)

    def _draw_face_template_model(self, face: Face) -> None:
        if not self.show_template_model:
            return
        self.visualizer.draw_3d_points(face.model3d,
                                       color=(255, 0, 525),
                                       size=1)

    def _display_normalized_image(self, face: Face) -> None:
        if not self.config.demo.display_on_screen:
            return
        if not self.show_normalized_image:
            return
        if self.config.mode == GazeEstimationMethod.MPIIGaze.name:
            reye = face.reye.normalized_image
            leye = face.leye.normalized_image
            normalized = np.hstack([reye, leye])
        elif self.config.mode == GazeEstimationMethod.MPIIFaceGaze.name:
            normalized = face.normalized_image
        else:
            raise ValueError
        if self.config.demo.use_camera:
            normalized = normalized[:, ::-1]
        cv2.imshow('normalized', normalized)

    def _draw_gaze_vector(self, face: Face) -> None:
        length = self.config.demo.gaze_visualization_length
        if self.config.mode == GazeEstimationMethod.MPIIGaze.name:
            for key in [FacePartsName.REYE, FacePartsName.LEYE]:
                eye = getattr(face, key.name.lower())
                self.visualizer.draw_3d_line(
                    eye.center, eye.center + length * eye.gaze_vector)
                pitch, yaw = np.rad2deg(eye.vector_to_angle(eye.gaze_vector))
                logger.info(
                    f'[{key.name.lower()}] pitch: {pitch:.2f}, yaw: {yaw:.2f}')
        elif self.config.mode == GazeEstimationMethod.MPIIFaceGaze.name:
            self.visualizer.draw_3d_line(
                face.center, face.center + length * face.gaze_vector)
            pitch, yaw = np.rad2deg(face.vector_to_angle(face.gaze_vector))
            logger.info(f'[face] pitch: {pitch:.2f}, yaw: {yaw:.2f}')
        else:
            raise ValueError


    def _log_out(self):
        # 创建一个handler，用于写入日志文件
        if self.config.demo.image_path :
            rq = str(self.config.demo.image_path).split('/')[-2] + '_' + str(self.config.demo.image_path).split('/')[-1]
        elif self.config.demo.video_path :
            rq = str(self.config.demo.video_path).split('/')[-1]
        logfile = self.config.demo.output_dir + rq + '.log'
        fh = logging.FileHandler(logfile, mode='w')
        fh.setLevel(logging.DEBUG)  # 输出到file的log等级的开关
        # 定义handler的输出格式
        formatter = logging.Formatter("%(levelname)s: %(message)s")
        fh.setFormatter(formatter)
        # 第四步，将logger添加到handler里面
        logger.addHandler(fh)
