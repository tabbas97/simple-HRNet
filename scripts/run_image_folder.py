import os
import sys
import glob
import argparse
import ast
import cv2
import time
import torch
from vidgear.gears import CamGear
import numpy as np

sys.path.insert(1, os.getcwd())
from SimpleHRNet import SimpleHRNet
from misc.visualization import draw_points, draw_skeleton, draw_points_and_skeleton, joints_dict, check_video_rotation
from misc.utils import find_person_id_associations

def draw_boxes(frame, boxes):
    for box in boxes:         
        frame = cv2.rectangle(frame, (box[0], box[1]), ((box[2]),(box[3])), (127, 255, 63), 2)
        # frame = cv2.rectangle(frame, (box[0], box[2]), ((box[0]+box[1]),(box[2] + box[3])), (127, 255, 63), 2)
    return frame

def main(folderpath, hrnet_c, hrnet_j, hrnet_weights, hrnet_joints_set, image_resolution, single_person,
         disable_tracking, max_batch_size, disable_vidgear, save_video, video_format,
         video_framerate, device, out_path):
    if device is not None:
        device = torch.device(device)
    else:
        if torch.cuda.is_available() and True:
            torch.backends.cudnn.deterministic = True
            device = torch.device('cuda:0')
        else:
            device = torch.device('cpu')

    print(device)

    image_resolution = ast.literal_eval(image_resolution)
    has_display = 'DISPLAY' in os.environ.keys() or sys.platform == 'win32'
    video_writer = None

    if folderpath is not None:
        file_list = glob.glob(folderpath + "/*.jpg")
        assert len(file_list) > 0
    if out_path is None:
        if not os.path.exists(os.path.join(folderpath, "outputs")):
            os.mkdir(os.path.join(folderpath, "outputs"))
        out_path = os.path.join(folderpath, "outputs")
    else:
        if not os.path.exists(out_path):
            os.mkdir(out_path)

    model = SimpleHRNet(
        hrnet_c,
        hrnet_j,
        hrnet_weights,
        resolution=image_resolution,
        multiperson=not single_person,
        return_bounding_boxes=True,
        max_batch_size=max_batch_size,
        device=device
    )

    for file in file_list:
        print ("Running on : \t", file)
        t = time.time()
        if file is not None :
            frame = cv2.imread(file)
            if frame is None:
                break
        else:
            break

        boxes, pts = model.predict(frame)
        frame = draw_boxes(frame, boxes)

        person_ids = np.arange(len(pts), dtype=np.int32)

        for i, (pt, pid) in enumerate(zip(pts, person_ids)):
            frame = draw_points_and_skeleton(frame, pt, joints_dict()[hrnet_joints_set]['skeleton'], person_index=pid,
                                                 points_color_palette='gist_rainbow', skeleton_color_palette='jet',
                                                 points_palette_samples=10)

        fps = 1. / (time.time() - t)
        print('\rframerate: %f fps' % fps, end='\t')

        has_display = False
        if has_display:
            cv2.imshow('frame.png', frame)
            k = cv2.waitKey(1)
            cv2.imwrite(os.path.join(out_path, file.split("/")[-1]), frame)
            if k == 27:  # Esc button
                if disable_vidgear:
                    video.release()
                else:
                    video.stop()
                break
        else:
            cv2.imwrite(os.path.join(out_path, file.split("/")[-1]), frame)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--folderpath", "-f", help="open the specified video (overrides the --camera_id option)",
                        type=str, default=None)
    parser.add_argument("--hrnet_c", "-c", help="hrnet parameters - number of channels", type=int, default=32)
    parser.add_argument("--hrnet_j", "-j", help="hrnet parameters - number of joints", type=int, default=17)
    parser.add_argument("--hrnet_weights", "-w", help="hrnet parameters - path to the pretrained weights",
                        type=str, default="./weights/pose_hrnet_w32_384x288.pth")
    parser.add_argument("--hrnet_joints_set",
                        help="use the specified set of joints ('coco' and 'mpii' are currently supported)",
                        type=str, default="coco")
    parser.add_argument("--image_resolution", "-r", help="image resolution", type=str, default='(384, 288)')
    parser.add_argument("--single_person",
                        help="disable the multiperson detection (YOLOv3 or an equivalen detector is required for"
                             "multiperson detection)",
                        action="store_true", default=False) # We need the detector to run on all frames in images for internal evaluation purposes.
                        				    # Enable accordingly.
    parser.add_argument("--disable_tracking",
                        help="disable the skeleton tracking and temporal smoothing functionality",
                        action="store_true", default=False) # Do not enable. This option was present for video applications only.
    parser.add_argument("--max_batch_size", help="maximum batch size used for inference", type=int, default=16)
    parser.add_argument("--disable_vidgear",
                        help="disable vidgear (which is used for slightly better realtime performance)",
                        action="store_true", default=True)  # see https://pypi.org/project/vidgear/
                        				    # Disabled for compat purposes.
    parser.add_argument("--save_video", help="save output frames into a video.", action="store_true")
    parser.add_argument("--video_format", help="fourcc video format. Common formats: `MJPG`, `XVID`, `X264`."
                                                     "See http://www.fourcc.org/codecs.php", type=str, default='MJPG')
    parser.add_argument("--video_framerate", help="video framerate", type=float, default=30)
    parser.add_argument("--device", help="device to be used (default: cuda, if available)", type=str, default=None)
    parser.add_argument("--out_path", help="Path to place the output images into", type=str, default=None)
    args = parser.parse_args()
    main(**args.__dict__)
