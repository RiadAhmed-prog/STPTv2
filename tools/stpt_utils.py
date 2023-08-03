import copy

import torch
import cv2
import numpy as np
import math
import re
from mmengine.dataset import Compose

# colors bgr
blue = (255, 0, 0)
green = (0, 255, 0)
red = (0, 0, 255)
orange = (0, 165, 255)
yellow = (0, 255, 255)

coco_keypoints = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]


def load_missing_data(file):
    mising_data = []
    with open(file, "r") as file:
        for line in file:
            line = line.strip()
            mising_data.append(line)

    return mising_data


def display_frames(filename, frames):
    index = 0
    total_frames = len(frames)

    while True:
        frame = frames[index]
        if frame.shape[1] >= 1080:
            frame = cv2.resize(frame, dsize=(int(frame.shape[1] // 1.5), int(frame.shape[0] // 1.5)))
        cv2.imshow(filename, frame)

        key = cv2.waitKey(1) & 0xFF  # Use a small delay (1 millisecond)

        if key == ord('d'):
            index = min(index + 1, total_frames - 1)
        elif key == ord('a'):
            index = max(index - 1, 0)
        elif key == 27:  # Press 'Esc' key to exit
            break

        # Check if the window is closed
        if cv2.getWindowProperty(filename, cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()


def put_multiline_text(image, text, org, fontFace, fontScale, color, thickness, line_space):
    lines = text.split('\n')

    for i, line in enumerate(lines):
        cv2.putText(image, line, (org[0], org[1] + i * len(coco_keypoints)), fontFace, fontScale, color, thickness, cv2.LINE_AA)


def find_key_by_value(dictionary, value):
    for key, val in dictionary.items():
        if val == value:
            return key
    return None  # Return None if the value is not found in the dictionary


def is_list_in_dict_keys(dictionary, integer_list):
    for num in integer_list:
        if num not in dictionary:
            return False
    return True


def load_label_map(file_path):
    """Load Label Map.

    Args:
        file_path (str): The file path of label map.

    Returns:
        dict: The label map (int -> label name).
    """
    lines = open(file_path).readlines()
    lines = [x.strip() for x in lines]
    classes = {}
    for line in lines:
        label = int(re.findall(r'\d+', line.split(': ')[0])[0]) - 1
        classes[label] = line.split(': ')[1]

    return classes


def plot_skeleton_kpts(im, kpts, steps, orig_shape=None):
    # Plot the skeleton and keypointsfor coco datatset
    palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
                        [230, 230, 0], [255, 153, 255], [153, 204, 255],
                        [255, 102, 255], [255, 51, 255], [102, 178, 255],
                        [51, 153, 255], [255, 153, 153], [255, 102, 102],
                        [255, 51, 51], [153, 255, 153], [102, 255, 102],
                        [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0],
                        [255, 255, 255]])

    skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
                [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
                [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

    pose_limb_color = palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]
    pose_kpt_color = palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]
    radius = 3
    num_kpts = len(kpts) // steps

    for kid in range(num_kpts):
        r, g, b = pose_kpt_color[kid]
        x_coord, y_coord = kpts[steps * kid], kpts[steps * kid + 1]
        if not (x_coord % 640 == 0 or y_coord % 640 == 0):
            if steps == 3:
                conf = kpts[steps * kid + 2]
                if conf < 0.5:
                    continue
            cv2.circle(im, (int(x_coord), int(y_coord)), radius, (int(r), int(g), int(b)), -1)
            # cv2.putText(im, str(kid), (int(x_coord+1), int(y_coord+1)), cv2.FONT_HERSHEY_COMPLEX,
            #             1, (255,255,255), 1, cv2.LINE_AA)

    for sk_id, sk in enumerate(skeleton):
        r, g, b = pose_limb_color[sk_id]
        pos1 = (int(kpts[(sk[0] - 1) * steps]), int(kpts[(sk[0] - 1) * steps + 1]))
        pos2 = (int(kpts[(sk[1] - 1) * steps]), int(kpts[(sk[1] - 1) * steps + 1]))
        if steps == 3:
            conf1 = kpts[(sk[0] - 1) * steps + 2]
            conf2 = kpts[(sk[1] - 1) * steps + 2]
            if conf1 < 0.5 or conf2 < 0.5:
                continue
        if pos1[0] % 640 == 0 or pos1[1] % 640 == 0 or pos1[0] < 0 or pos1[1] < 0:
            continue
        if pos2[0] % 640 == 0 or pos2[1] % 640 == 0 or pos2[0] < 0 or pos2[1] < 0:
            continue
        cv2.line(im, pos1, pos2, (int(r), int(g), int(b)), thickness=1)


# TODO: implement visualizing features
def plot_skeleton_features(im, kpts, fpts, steps=3, feature: str = None):
    coords = {
        'velocity': ['vel', 0],
        'acceleration': ['acc', 1],
        'impact force': ['IF', 2]
    }
    # Plot the skeleton and keypointsfor coco datatset
    palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
                        [230, 230, 0], [255, 153, 255], [153, 204, 255],
                        [255, 102, 255], [255, 51, 255], [102, 178, 255],
                        [51, 153, 255], [255, 153, 153], [255, 102, 102],
                        [255, 51, 51], [153, 255, 153], [102, 255, 102],
                        [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0],
                        [255, 255, 255]])

    skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
                [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
                [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

    pose_limb_color = palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]
    pose_kpt_color = palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]
    num_kpts = len(kpts) // steps
    if feature == 'rgb-w':
        for kid in range(num_kpts):
            r, g, b, radius = fpts[steps * kid], fpts[steps * kid + 1], fpts[steps * kid + 2], fpts[steps * kid + 3]
            x_coord, y_coord = kpts[steps * kid], kpts[steps * kid + 1]
            if not (x_coord % 640 == 0 or y_coord % 640 == 0):
                conf = kpts[steps * kid + steps - 1]
                if conf < 0.5:
                    continue
                cv2.circle(im, (int(x_coord), int(y_coord)), radius, (int(r), int(g), int(b)), -1)
                # cv2.putText(im, str(kid), (int(x_coord+1), int(y_coord+1)), cv2.FONT_HERSHEY_COMPLEX,
                #             1, (255,255,255), 1, cv2.LINE_AA)

        # TODO: implement visualizing limb rgb-w feature
        for sk_id, sk in enumerate(skeleton):
            continue

    else:
        formatted_string = f"{coords[feature][0]}---\n"
        for kid in range(num_kpts):
            r, g, b = pose_kpt_color[kid]
            x_coord, y_coord = kpts[steps * kid], kpts[steps * kid + 1]
            x_feature, y_feature = fpts[steps * kid], kpts[steps * kid + 1]
            if not (x_coord % 640 == 0 or y_coord % 640 == 0):
                conf = kpts[steps * kid + steps - 1]
                if conf < 0.5:
                    continue

                # cv2.circle(im, (int(x_coord), int(y_coord)), radius, (int(r), int(g), int(b)), -1)
                formatted_string += f"{coco_keypoints[kid]} - {x_feature, y_feature}\n"
        put_multiline_text(im, formatted_string, (int(coords[feature][1] * 300), int(20)),
                           cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # TODO: implement visualizing limb feature
        for sk_id, sk in enumerate(skeleton):
            continue


def drawline_from_point_list(image, points: np.ndarray):
    """
    image: black image to draw STPT plot on
    points: keypoints of shape (M, T, V, C) -->
            M = no. of person, T = no. of frames, V, C = (17, 2) for coco keypoints
    """
    # Line thickness of 1 px
    thickness = 1

    for f_idx in range(points.shape[1]):
        for p_idx in range(points.shape[0]):
            if f_idx == 0:
                start = points[p_idx, f_idx]
            else:
                end = points[p_idx, f_idx]

                for start_point, end_point in zip(start, end):
                    # if both increase
                    if end_point[0] > start_point[0] and end_point[1] > start_point[1]:
                        color = red
                    # if x increases
                    elif end_point[0] > start_point[0] and end_point[1] <= start_point[1]:
                        color = green
                    # if y increases
                    elif end_point[0] <= start_point[0] and end_point[1] > start_point[1]:
                        color = orange
                    # if both decrease
                    else:
                        color = yellow

                    image = cv2.line(image, (int(start_point[0]), int(start_point[1])),
                                     (int(end_point[0]), int(end_point[1])), color, thickness)

                    # current end point is next start point
                start = end

    return image


def drawline_from_point_list_stgcn(image, points, ranked_kp: list):
    """
    image: black image to draw STPT plot on
    points: keypoints of shape (M, T, V, C) -->
            M = no. of person, T = no. of frames, V, C = (17, 2) for coco keypoints
    ranked_kp: important keypoints output from STGCN model
    """
    # Line thickness of 1 px

    for f_idx in range(points.shape[1]):
        for p_idx in range(points.shape[0]):
            if f_idx == 0:
                start = points[p_idx, f_idx]
            else:
                end = points[p_idx, f_idx]

                for j, (start_point, end_point) in enumerate(zip(start, end)):
                    # if both increase
                    if end_point[0] > start_point[0] and end_point[1] > start_point[1]:
                        color = red
                    # if x increases
                    elif end_point[0] > start_point[0] and end_point[1] <= start_point[1]:
                        color = green
                    # if y increases
                    elif end_point[0] <= start_point[0] and end_point[1] > start_point[1]:
                        color = orange
                    # if both decrease
                    else:
                        color = yellow
                    if j == ranked_kp[0]:
                        thickness = 6
                    elif j == ranked_kp[1]:
                        thickness = 2
                    else:
                        thickness = 1

                    image = cv2.line(image, (int(start_point[0]), int(start_point[1])),
                                     (int(end_point[0]), int(end_point[1])), color, thickness)

                    # current end point is next start point
                start = end

    return image


def calculate_features(annotation, feature: str):
    if feature not in ['velocity', 'acceleration', 'impact force', 'rgb-w']:
        print(NotImplementedError(f"{feature} not implemented yet"))
        return None

    keypoints = np.concatenate((annotation['keypoint'], np.expand_dims(annotation['keypoint_score'], axis=-1)), axis=-1)
    if feature == 'rgb-w':
        shape = (keypoints.shape[0], keypoints.shape[1], keypoints.shape[2], 4)
    else:
        shape = keypoints.shape
    feature_points = np.zeros(shape)

    # TODO: implement feature
    for f_idx in range(keypoints.shape[1]):
        for p_idx in range(keypoints.shape[0]):
            if f_idx == 0:
                start = keypoints[p_idx, f_idx]
            else:
                end = keypoints[p_idx, f_idx]
                if feature == 'velocity':
                    # TODO: derive formula
                    for i, (start_point, end_point) in enumerate(zip(start, end)):
                        feature_points[p_idx, f_idx, i][0] = end_point[0] - start_point[0]
                if feature == 'acceleration':
                    # TODO: derive formula
                    for i, (start_point, end_point) in enumerate(zip(start, end)):
                        feature_points[p_idx, f_idx, i][0] = end_point[0] - start_point[0]
                if feature == 'impact force':
                    # TODO: derive formula
                    for i, (start_point, end_point) in enumerate(zip(start, end)):
                        feature_points[p_idx, f_idx, i][0] = end_point[0] - start_point[0]
                if feature == 'rgb-w':
                    # TODO: derive formula
                    pass

                start = end

    return feature_points


def colouring_halt_points(image, points):
    """
    image: black image to draw STPT plot on
    points: keypoints of shape (M, T, V, C) -->
            M = no. of person, T = no. of frames, V, C = (17, 2) for coco keypoints
    """
    # Blue color in RGB
    color = blue
    thresh = 5
    radius = 4
    x_difference = 0
    y_difference = 0
    total_difference = 0
    # thickness of -1 px for filled circle
    thickness = -1
    for person in points:
        for i, frame in enumerate(person):
            if i == 0:
                start = frame
            else:
                if i <= 3:
                    end = frame

                    for start_point, end_point in zip(start, end):
                        x_difference = x_difference + abs(end_point[0] - start_point[0])
                        y_difference = y_difference + abs(end_point[1] - start_point[1])
                        total_difference = x_difference + y_difference
                        if total_difference <= thresh:
                            image = cv2.circle(image, (int(start_point[0]), int(start_point[1])), radius, color,
                                               thickness)
                            image = cv2.circle(image, (int(end_point[0]), int(end_point[1])), radius, color, thickness)

                    start = end

                else:
                    x_difference = 0
                    y_difference = 0

    return image


def colouring_halt_points_wrong(image, points):
    """
    image: black image to draw STPT plot on
    points: keypoints of shape (M, T, V, C) -->
            M = no. of person, T = no. of frames, V, C = (17, 2) for coco keypoints
    """
    # Blue color in RGB
    color = blue
    count = 0
    thresh = 5
    radius = 4
    x_difference = 0
    y_difference = 0
    total_difference = 0
    # thickness of -1 px for filled circle
    thickness = -1
    for person in points:
        for frame in person:
            for i, kp in enumerate(frame):
                if i == 0:
                    start_point = kp
                else:
                    if i <= 3:
                        end_point = kp
                        x_difference = x_difference + abs(end_point[0] - start_point[0])
                        y_difference = y_difference + abs(end_point[1] - start_point[1])
                        total_difference = x_difference + y_difference
                        if total_difference <= thresh:
                            image = cv2.circle(image, (int(start_point[0]), int(start_point[1])), radius, color,
                                               thickness)
                            image = cv2.circle(image, (int(end_point[0]), int(end_point[1])), radius, color, thickness)
                        start_point = end_point
                    else:
                        x_difference = 0
                        y_difference = 0

    return image


def generate_frame_plot(annotation, output_path=None):
    keypoints = annotation['keypoint']
    shape = (annotation['img_shape'][0], annotation['img_shape'][1], 3)

    img = np.zeros(shape, dtype=np.uint8)

    img = drawline_from_point_list(img, keypoints)
    img = colouring_halt_points(img, keypoints)

    if output_path is not None:
        cv2.imwrite(output_path, img)

    return img


def generate_frame_plot_kpts(annotation, output_path=None):
    keypoints = np.concatenate((annotation['keypoint'], np.expand_dims(annotation['keypoint_score'], axis=-1)), axis=-1)
    shape = (annotation['img_shape'][0], annotation['img_shape'][1], 3)

    img = np.zeros(shape, dtype=np.uint8)

    for f_idx in range(keypoints.shape[1]):
        for p_idx in range(keypoints.shape[0]):
            plot_skeleton_kpts(img, keypoints[p_idx, f_idx].flatten().tolist(), steps=3)

    if output_path is not None:
        cv2.imwrite(output_path, img)

    return img


def generate_frame_plot_stgcn(cfg, model, device, annotation, output_path=None):
    keypoints = annotation['keypoint']
    shape = (annotation['img_shape'][0], annotation['img_shape'][1], 3)

    img = np.zeros(shape, dtype=np.uint8)
    pipeline = Compose(cfg.val_pipeline)
    ann = copy.deepcopy(annotation)
    packed_results = pipeline(ann)
    inputs = packed_results['inputs'].unsqueeze(0).to(device)
    feats = model(inputs, mode='tensor')
    sum_intensity = torch.sum(feats, dim=(1, 2, 3))
    ranked_kp = np.argsort(sum_intensity.cpu().detach().numpy()).squeeze()

    img = drawline_from_point_list_stgcn(img, keypoints, ranked_kp)
    img = colouring_halt_points(img, keypoints)

    if output_path is not None:
        cv2.imwrite(output_path, img)

    return img


def generate_video(annotation, output_filename=None, show=False, features: list = None):
    keypoints = np.concatenate((annotation['keypoint'], np.expand_dims(annotation['keypoint_score'], axis=-1)), axis=-1)
    shape = (annotation['img_shape'][0], annotation['img_shape'][1], 3)

    frames = []
    for _ in range(annotation['total_frames']):
        frames.append(np.zeros(shape, dtype=np.uint8))

    for f_idx in range(keypoints.shape[1]):
        for p_idx in range(keypoints.shape[0]):
            kp = keypoints[p_idx, f_idx][:17, :]
            plot_skeleton_kpts(frames[f_idx], kp.flatten().tolist(), steps=3)

    if features is not None:
        for feature in features:
            steps = 4 if feature == 'rgb-w' else 3
            feature_points = annotation[feature]
            for f_idx in range(keypoints.shape[1]):
                for p_idx in range(keypoints.shape[0]):
                    kp = keypoints[p_idx, f_idx][:17, :]
                    points = feature_points[p_idx, f_idx][:17, :]
                    plot_skeleton_features(frames[f_idx], kp.flatten().tolist(), points.flatten().tolist(),
                                           steps=steps, feature=feature)

    if show:
        display_frames(annotation['frame_dir'], frames)

    if output_filename is not None:
        print(output_filename)
        fps = 15
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        vid = cv2.VideoWriter(output_filename, fourcc, fps, (shape[1], shape[0]))
        for frame in frames:
            vid.write(frame)

        vid.release()
