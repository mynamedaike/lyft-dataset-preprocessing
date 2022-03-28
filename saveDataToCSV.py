import os
import numpy as np
from tqdm import tqdm

from l5kit.configs import load_config_data
from l5kit.data import LocalDataManager, ChunkedDataset, PERCEPTION_LABELS
from l5kit.dataset import AgentDataset, EgoDataset
from l5kit.rasterization import build_rasterizer
from l5kit.geometry import transform_point

# set env variable for data
os.environ["L5KIT_DATA_FOLDER"] = "./sample"
dm = LocalDataManager(None)
# get config
cfg = load_config_data("./agent_motion_config.yaml")

train_cfg = cfg["train_data_loader"]
rasterizer = build_rasterizer(cfg, dm)
zarr_dataset = ChunkedDataset(dm.require(train_cfg["key"])).open()
ego_dataset = EgoDataset(cfg, zarr_dataset, rasterizer)
agent_dataset = AgentDataset(cfg, zarr_dataset, rasterizer)

DATA_TIME_STEP = 0.1
TRAIN_TIME_STEP = 0.5
AGENT_LABEL = {1: 'UNKNOWN', 3: 'VEHICLE', 12: 'CYCLIST', 14: 'PEDESTRIAN'}
LEBEL_CODE = {'VEHICLE': 0, 'PEDESTRIAN': 1, 'CYCLIST': 2}

scene_num = len(ego_dataset.dataset.scenes)
step_frames = int(TRAIN_TIME_STEP / DATA_TIME_STEP)
print('The number of Scenes:', scene_num)
print('Step Frames:', step_frames)

def getAgentData(index, agent, isEgo = True):
    # fix me
    # filter agents whose label are unknown and that do not have future trajectories
    if agent['label_index'] == 1 or (isEgo and agent['target_availabilities'][1] == 0) or (
            not isEgo and agent['target_availabilities'][1] == 0):
        return None
    data = []

    data.append(index)
    data.append(agent['track_id'])
    data.append(agent['centroid'][0])
    data.append(agent['centroid'][1])
    data.append(agent['extent'][1])
    data.append(agent['extent'][0])
    data.append(agent['yaw'])
    data.append(LEBEL_CODE[AGENT_LABEL[agent['label_index']]])
    data.append(index)
    data.append(int(agent['timestamp'] / 1000000))

    # fix me
    position0 = transform_point(agent['target_positions'][0], agent['world_from_agent'])
    position1 = transform_point(agent['target_positions'][1], agent['world_from_agent'])

    if isEgo:
        velocity = (position0 - agent['centroid']) / DATA_TIME_STEP
    else:
        velocity = agent['velocity']

    velocity = (velocity[0] ** 2 + velocity[1] ** 2) ** 0.5

    next_velocity = (position1 - position0) / DATA_TIME_STEP
    acceleration = (next_velocity - velocity) / DATA_TIME_STEP
    acceleration = (acceleration[0] ** 2 + acceleration[1] ** 2) ** 0.5
    yaw_rate = agent['target_yaws'][0][0] / DATA_TIME_STEP
    scene_index = agent['scene_index']

    data.append(velocity)
    data.append(acceleration)
    data.append(yaw_rate)
    data.append(scene_index)

    return data


for scene_index in tqdm(range(scene_num)):
    dataset_to_save = []
    frame_indices = ego_dataset.get_scene_indices(scene_index)

    for i, absolute_frame_index in enumerate(range(frame_indices[0], frame_indices[-1], step_frames)):
        agent = ego_dataset[absolute_frame_index]
        data = getAgentData(i, agent)
        if data is not None:
            dataset_to_save.append(data)

        agent_indices = agent_dataset.get_frame_indices(absolute_frame_index)
        if len(agent_indices) == 0:
            continue

        for agent_index in agent_indices:
            agent = agent_dataset[agent_index]
            data = getAgentData(i, agent, False)
            if data is not None:
                dataset_to_save.append(data)

    dataset_to_save = np.array(dataset_to_save, dtype = np.float64)

    dir_name = './dataset'
    file_name = 'agent_dataset_scene' + str(scene_index) + '.csv'
    np.savetxt(os.path.join(dir_name, file_name), dataset_to_save, delimiter = ',')
