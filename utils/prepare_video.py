import argparse
import json
import os
import cv2


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--skip', type=int, default=10)
    parser.add_argument('video_dir_path')
    parser.add_argument('scene_dir_path')

    return parser.parse_args()


def process_videos(args):
    video_dir_path = args.video_dir_path
    scene_dir_path = args.scene_dir_path
    skip = args.skip

    videos = [x for x in os.listdir(video_dir_path) if '.m4v' in x.lower()]

    for video in videos:
        scene_name = video.split('.')[0]

        scene_dir = os.path.join(scene_dir_path, scene_name)
        os.makedirs(scene_dir, exist_ok=True)

        cap = cv2.VideoCapture(os.path.join(video_dir_path, video))
        total_frames = 0
        ret = True
        img_names = []


        while ret:
            ret, frame = cap.read()
            if ret:
                img_name = f'{total_frames:08d}.jpg'
                img_names.append(img_name)
                width = frame.shape[1]
                heigth = frame.shape[0]
                cv2.imwrite(os.path.join(scene_dir, img_name), frame)
                total_frames += 1

        lines = []
        for i in range(0, total_frames - 2 * skip):
            lines.append(f'{i:08d}.jpg {i + skip:08d}.jpg {i + 2*skip:08d}.jpg\n')

        with open(os.path.join(scene_dir, 'triplets.txt'), 'w') as f:
            f.writelines(lines)

        pp = [width / 2, heigth / 2]

        focal = 1610 if width == 1920 else 1073
        K = [[focal, 0, pp[0]], [0, focal, pp[1]], [0, 0, 1]]

        camera = {'K': K, 'fx': focal, 'fy': focal, 'pp': pp, 'width': width, 'height': heigth}
        cameras = {"phone_0": camera}
        images = {name: "phone_0" for name in img_names}

        with open(os.path.join(scene_dir, 'calibration.json'), 'w') as f:
            json.dump({'cameras': cameras, 'images': images}, f, indent=2)


if __name__ == '__main__':
    args = parse_args()
    process_videos(args)