import os
import random

train_camera_suffix = [f'_{str(x).zfill(2)}' for x in range(5)]
trg_cam, src_cam = random.sample(train_camera_suffix, 2)

print(trg_cam, src_cam)
