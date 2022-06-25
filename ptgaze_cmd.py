import subprocess
import psutil
import time
import argparse


def isRuningPid(pid):
    try:
        s = psutil.Process(pid)
        return True
    except psutil.NoSuchProcess:
        return False


def parse():
    parser = argparse.ArgumentParser(description='arguments of script')
    parser.add_argument('--runningPid', default=0, type=int, metavar='P',
                        help='run the program after the pid is over')

    args = parser.parse_args()

    return args


args = parse()
pid = args.runningPid


def main():
    linux_commod = [
        'python  main.py --device=cpu --image=E:\Project\child_eyetrace\Data\img2_group0\lmz --face-detector=mtcnn --output-dir=E:\Project\child_eyetrace\gaze_result2\group0',
        'python  main.py --device=cpu --image=E:\Project\child_eyetrace\Data\img2_group0\lyl --face-detector=mtcnn --output-dir=E:\Project\child_eyetrace\gaze_result2\group0',


        ]

    while True:
        #time.sleep(1)
        if isRuningPid(pid) == False:

            for i_commod in linux_commod:
                print(i_commod)
                result = subprocess.getstatusoutput(i_commod)
                print(result[1])

            break


if __name__ == "__main__":
    main()