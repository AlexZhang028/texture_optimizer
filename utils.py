import numpy as np


def load_kitti_poses(file_path):
    poses = []
    with open(file_path, 'r') as file:
        for line in file:
            values = list(map(float, line.strip().split()))
            pose = np.array(values).reshape(3, 4)
            pose = np.vstack((pose, [0, 0, 0, 1]))
            poses.append(pose)
    return np.array(poses)