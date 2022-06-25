import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import cv2
import argparse
import logging
import time

from ptgaze import get_default_config
from ptgaze.demo import Demo
from ptgaze.utils import update_default_config, update_config


logger = logging.getLogger(__name__)


def video_cap():
    cap = cv2.VideoCapture(0)

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out = cv2.VideoWriter('E:/pytorch_mpiigaze_demo/Data/display.mp4', fourcc, 30.0, (640, 480))

    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            frame = cv2.flip(frame, -1)
            #        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))   将视频转换为灰色的源
            out.write(frame)

            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        type=str,
        help='Config file for YACS. When using a config file, all the other '
        'commandline arguments are ignored. '
        'See https://github.com/hysts/pytorch_mpiigaze_demo/configs/demo_mpiigaze.yaml'
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='eye',
        choices=['eye', 'face'],
        help='With \'eye\', MPIIGaze model will be used. With \'face\', '
        'MPIIFaceGaze model will be used. (default: \'eye\')')
    parser.add_argument(
        '--face-detector',
        type=str,
        default='dlib',
        choices=['dlib', 'face_alignment_dlib', 'face_alignment_sfd'],
        help='The method used to detect faces and find face landmarks '
        '(default: \'dlib\')')
    parser.add_argument('--device',
                        type=str,
                        choices=['cpu', 'cuda'],
                        default='cpu',
                        help='Device used for model inference.')
    parser.add_argument('--image',
                        type=str,
                        help='Path to an input image file.')
    parser.add_argument('--video',
                        type=str,
                        default='E:/pytorch_mpiigaze_demo/Data/display.mp4',
                        help='Path to an input video file.')
    parser.add_argument(
        '--camera',
        type=str,
        help='Camera calibration file. '
        'See https://github.com/hysts/pytorch_mpiigaze_demo/ptgaze/data/calib/sample_params.yaml'
    )
    parser.add_argument(
        '--output-dir',
        '-o',
        type=str,
        default='./result',
        help='If specified, the overlaid video will be saved to this directory.'
    )
    parser.add_argument('--ext',
                        '-e',
                        type=str,
                        choices=['avi', 'mp4'],
                        default='mp4',
                        help='Output video file extension.')
    parser.add_argument(
        '--no-screen',
        action='store_true',
        help='If specified, the video is not displayed on screen, and saved '
        'to the output directory.')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    if args.debug:
        logging.getLogger('ptgaze').setLevel(logging.DEBUG)


    config = get_default_config()
    if args.config:
        config.merge_from_file(args.config)
        if (args.device or args.image or args.video or args.camera
                or args.output_dir or args.ext or args.no_screen):
            raise RuntimeError(
                'When using a config file, all the other commandline '
                'arguments are ignored.')
        if config.demo.image_path and config.demo.video_path:
            raise ValueError(
                'Only one of config.demo.image_path or config.demo.video_path '
                'can be specified.')
    else:
        update_default_config(config, args)


    update_config(config)
    logger.info(config)

    demo = Demo(config)
    demo.run()

if __name__ == '__main__':
    video_cap()
    main()

