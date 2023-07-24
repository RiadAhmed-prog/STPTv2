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


def drawline_from_point_list(image, points):
    """
    image: black image to draw STPT plot on
    points: keypoints of shape (M, T, V, C) -->
            M = no. of person, T = no. of frames, V, C = (17, 2) for coco keypoints
    """
    # Line thickness of 1 px
    thickness = 1

    for person in points:
        for i, frame in enumerate(person):
            if i == 0:
                start = frame
            else:
                end = frame

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


def drawline_from_point_list_wrong(image, points):
    """
    image: black image to draw STPT plot on
    points: keypoints of shape (M, T, V, C) -->
            M = no. of person, T = no. of frames, V, C = (17, 2) for coco keypoints
    """
    count = 0
    # Line thickness of 1 px
    thickness = 1

    for person in points:
        for frame in person:
            for i, kp in enumerate(frame):
                if i == 0:
                    start_point = kp
                else:
                    end_point = kp

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
                    start_point = end_point
    return image


def drawline_from_point_list_stgcn(image, points, ranked_kp: list):
    """
    image: black image to draw STPT plot on
    points: keypoints of shape (M, T, V, C) -->
            M = no. of person, T = no. of frames, V, C = (17, 2) for coco keypoints
    ranked_kp: important keypoints output from STGCN model
    """
    # Line thickness of 1 px

    for person in points:
        for i, frame in enumerate(person):
            if i == 0:
                start = frame
            else:
                end = frame

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


def calc_vel_acc(points):
    vel = []
    acc = []
    time = 5
    prev_vel_X = 0
    prev_vel_Y = 0
    for count, point in enumerate(points):
        if count > 0 and count % 5 == 0:
            vel_X = (points[count][0] - points[count - 5][0]) / time
            vel_Y = (points[count][1] - points[count - 5][1]) / time
            vel_lobdhi = math.sqrt(pow(vel_X, 2) + pow(vel_Y, 2))
            vel.append(vel_lobdhi)

            acc_X = (vel_X - prev_vel_X) / time
            acc_Y = (vel_Y - prev_vel_Y) / time
            prev_vel_X = vel_X
            prev_vel_Y = vel_Y
            acc.append((round(acc_X, 3), round(acc_Y, 3)))

    print("Velocity:", max(vel))
    # print(round(acc))


def detect_dir(points):
    directions = []
    prev_X = 0
    prev_Y = 0
    for point in points:
        if point[0] > prev_X:
            dir_X = 1
        else:
            dir_X = -1
        if point[1] > prev_Y:
            dir_Y = 1
        else:
            dir_Y = -1
        prev_X = point[0]
        prev_Y = point[1]

        directions.append((dir_X, dir_Y))

    print(directions)


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


def generate_frame_plot(annotation, output_path):
    important_parts = ["head", "abdomen", "chest", "l_hand", "l_elbow", "l_shoulder", "r_hand", "r_elbow",
                       "r_shoulder"]
    keypoints = annotation['keypoint']
    shape = (annotation['img_shape'][0], annotation['img_shape'][1], 3)

    img = np.zeros(shape, dtype=np.uint8)

    img = drawline_from_point_list(img, keypoints)
    img = colouring_halt_points(img, keypoints)

    cv2.imwrite(output_path, img)


def generate_frame_plot_stgcn(cfg, model, device, annotation, output_path):
    important_parts = ["head", "abdomen", "chest", "l_hand", "l_elbow", "l_shoulder", "r_hand", "r_elbow",
                       "r_shoulder"]
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

    cv2.imwrite(output_path, img)


def generate_kp_video(annotation, output_filename):
    keypoints = annotation['keypoint']
    keypoints_score = annotation['keypoint_score'][..., np.newaxis]
    keypoints_concat = np.concatenate([keypoints, keypoints_score], axis=-1)

    fps = 10
    shape = (annotation['img_shape'][0], annotation['img_shape'][1], 3)

    frames = []
    for _ in range(annotation['total_frames']):
        frames.append(np.zeros(shape, dtype=np.uint8))

    for person in keypoints_concat:
        start_index = 0
        if len(frames) > person.shape[0]:
            start_index = len(frames) - person.shape[0]

        for i, kp in enumerate(person):
            frame = frames[i + start_index]
            plot_skeleton_kpts(frame, kp.reshape(51), 3)
            frames[i + start_index] = frame

    vid = cv2.VideoWriter(output_filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, (shape[1], shape[0]))
    for frame in frames:
        vid.write(frame)

    vid.release()
