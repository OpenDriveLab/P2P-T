import os
import fnmatch
from pathlib import Path

import h5py
import yaml
import numpy as np

from configs.state_vec import STATE_VEC_IDX_MAPPING


DATASET_INSTRUCTION = "Use the hammer to hit the target object."
DATASET_NAME = "hammer"
HDF5_DIR = "data/datasets/hammer"


class UR10eHDF5VLADataset:
    """
    This class is used to sample episodes from the embododiment dataset
    stored in HDF5.
    """
    def __init__(self, dataset_dir: str | None = None) -> None:
        # [Modify] The path to the HDF5 dataset directory
        # Each HDF5 file contains one episode
        self.DATASET_NAME = DATASET_NAME
        hdf5_dir = self._resolve_dataset_dir(dataset_dir)
        
        self.file_paths = []
        for root, _, files in os.walk(hdf5_dir):
            for filename in fnmatch.filter(files, '*.hdf5') + fnmatch.filter(files, '*.h5'):
                file_path = os.path.join(root, filename)
                self.file_paths.append(file_path)
        self.file_paths = sorted(self.file_paths)
        if not self.file_paths:
            raise FileNotFoundError(
                f"No .hdf5/.h5 files found under {hdf5_dir}. "
                "Set HDF5_DIR in data/ur10e_hdf5_vla_dataset.py to the directory "
                "containing one HDF5 episode per file."
            )
                
        # Load the config
        config_path = Path(__file__).resolve().parents[1] / "configs" / "ur10e_p2p_t.yaml"
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        self.CHUNK_SIZE = config['common']['action_chunk_size']
        self.IMG_HISORY_SIZE = config['common']['img_history_size']
        self.STATE_DIM = config['common']['state_dim']
        self.MIN_EPISODE_LEN = config['dataset'].get('epsd_len_thresh_low', 32)
    
        # Get each episode's len
        valid_file_paths = []
        episode_lens = []
        invalid_reasons = []
        for file_path in self.file_paths:
            try:
                valid, res = self.parse_hdf5_file_state_only(file_path)
            except Exception as exc:
                invalid_reasons.append(f"{file_path}: {exc}")
                continue
            _len = res['state'].shape[0] if valid else 0
            if _len > 0:
                valid_file_paths.append(file_path)
                episode_lens.append(_len)
            else:
                invalid_reasons.append(
                    f"{file_path}: shorter than epsd_len_thresh_low={self.MIN_EPISODE_LEN}"
                )
        self.file_paths = valid_file_paths
        episode_lens = np.array(episode_lens)
        total_len = np.sum(episode_lens)
        if total_len <= 0:
            details = "\n".join(invalid_reasons[:10])
            raise ValueError(
                f"No valid episodes found under {hdf5_dir}. "
                f"Need at least one episode with length >= {self.MIN_EPISODE_LEN}.\n"
                f"First invalid files:\n{details}"
            )
        self.episode_sample_weights = episode_lens / total_len

    def _resolve_dataset_dir(self, dataset_dir: str | None = None):
        repo_root = Path(__file__).resolve().parents[1]
        configured_dir = Path(
            dataset_dir or os.environ.get("P2P_T_DATA_ROOT", HDF5_DIR)
        ).expanduser()
        candidates = [
            configured_dir,
            repo_root / configured_dir,
        ]
        for candidate in candidates:
            if candidate.is_dir():
                return str(candidate)
        return str(candidates[0])
    
    def __len__(self):
        return len(self.file_paths)
    
    def get_dataset_name(self):
        return self.DATASET_NAME
    
    def get_item(self, index: int=None, state_only=False):
        """Get a training sample at a random timestep.

        Args:
            index (int, optional): the index of the episode.
                If not provided, a random episode will be selected.
            state_only (bool, optional): Whether to return only the state.
                In this way, the sample will contain a complete trajectory rather
                than a single timestep. Defaults to False.

        Returns:
           sample (dict): a dictionary containing the training sample.
        """
        while True:
            if index is None:
                file_path = np.random.choice(self.file_paths, p=self.episode_sample_weights)
            else:
                file_path = self.file_paths[index]
            valid, sample = self.parse_hdf5_file(file_path) \
                if not state_only else self.parse_hdf5_file_state_only(file_path)
            if valid:
                return sample
            else:
                index = np.random.randint(0, len(self.file_paths))
    
    def parse_hdf5_file(self, file_path):
        """[Modify] Parse a hdf5 file to generate a training sample at
            a random timestep.

        Args:
            file_path (str): the path to the hdf5 file
        
        Returns:
            valid (bool): whether the episode is valid, which is useful for filtering.
                If False, this episode will be dropped.
            dict: a dictionary containing the training sample,
                {
                    "meta": {
                        "dataset_name": str,    # the name of your dataset.
                        "#steps": int,          # the number of steps in the episode,
                                                # also the total timesteps.
                        "instruction": str      # the language instruction for this episode.
                    },                           
                    "step_id": int,             # the index of the sampled step,
                                                # also the timestep t.
                    "state": ndarray,           # state[t], (1, STATE_DIM).
                    "state_std": ndarray,       # std(state[:]), (STATE_DIM,).
                    "state_mean": ndarray,      # mean(state[:]), (STATE_DIM,).
                    "state_norm": ndarray,      # norm(state[:]), (STATE_DIM,).
                    "actions": ndarray,         # action[t:t+CHUNK_SIZE], (CHUNK_SIZE, STATE_DIM).
                    "state_indicator", ndarray, # indicates the validness of each dim, (STATE_DIM,).
                    "cam_high": ndarray,        # external camera image, (IMG_HISORY_SIZE, H, W, 3)
                                                # or (IMG_HISORY_SIZE, 0, 0, 0) if unavailable.
                    "cam_high_mask": ndarray,   # indicates the validness of each timestep, (IMG_HISORY_SIZE,) boolean array.
                                                # For the first IMAGE_HISTORY_SIZE-1 timesteps, the mask should be False.
                    "cam_left_wrist": ndarray,  # left wrist camera image, (IMG_HISORY_SIZE, H, W, 3).
                                                # or (IMG_HISORY_SIZE, 0, 0, 0) if unavailable.
                    "cam_left_wrist_mask": ndarray,
                    "cam_right_wrist": ndarray, # right wrist camera image, (IMG_HISORY_SIZE, H, W, 3).
                                                # or (IMG_HISORY_SIZE, 0, 0, 0) if unavailable.
                                                # If only one wrist, make it right wrist, plz.
                    "cam_right_wrist_mask": ndarray
                } or None if the episode is invalid.
        """
        with h5py.File(file_path, 'r') as f:
            obs = f['observations']
            qpos = obs['qpos'][:].astype(np.float32)
            actions_raw = f['actions'][:].astype(np.float32)
            num_steps = min(qpos.shape[0], actions_raw.shape[0], obs['rgb'].shape[0])
            qpos = qpos[:num_steps]
            actions_raw = actions_raw[:num_steps]
            # [Optional] We drop too-short episode
            if num_steps < self.MIN_EPISODE_LEN:
                return False, None
            
            # [Optional] We skip the first few still steps
            EPS = 1e-2
            # Get the idx of the first qpos whose delta exceeds the threshold
            qpos_delta = np.abs(qpos - qpos[0:1])
            indices = np.where(np.any(qpos_delta > EPS, axis=1))[0]
            if len(indices) > 0:
                first_idx = indices[0]
            else:
                first_idx = 0
            
            # We randomly sample a timestep
            start_idx = max(first_idx - 1, 0)
            step_id = np.random.randint(start_idx, num_steps)
            
            # Load the instruction
            instruction = DATASET_INSTRUCTION
            
            # Assemble the meta
            meta = {
                "dataset_name": self.DATASET_NAME,
                "#steps": num_steps,
                "step_id": step_id,
                "instruction": instruction,
                "instruction_text": instruction,
            }
            
            # Collected action rows are [6 arm joints, commanded gripper, observed gripper].
            # The model action/state space uses [6 arm joints, gripper].
            target_qpos = actions_raw[step_id:step_id+self.CHUNK_SIZE, :7]
            
            # Parse the state and action
            state = qpos[step_id:step_id+1]
            state_std = np.std(qpos, axis=0)
            state_mean = np.mean(qpos, axis=0)
            state_norm = np.sqrt(np.mean(qpos**2, axis=0))
            actions = target_qpos
            if actions.shape[0] < self.CHUNK_SIZE:
                # Pad the actions using the last action
                actions = np.concatenate([
                    actions,
                    np.tile(actions[-1:], (self.CHUNK_SIZE-actions.shape[0], 1))
                ], axis=0)
            
            # Fill the state/action into the unified vector
            def fill_in_state(values):
                # Target indices corresponding to your state space
                # knife data: 6 arm joints + 1 gripper.
                UNI_STATE_INDICES = [
                    STATE_VEC_IDX_MAPPING[f"arm_joint_{i}_pos"] for i in range(6)
                ] + [
                    STATE_VEC_IDX_MAPPING["gripper_open"]
                ]
                uni_vec = np.zeros(values.shape[:-1] + (self.STATE_DIM,), dtype=np.float32)
                uni_vec[..., UNI_STATE_INDICES] = values
                return uni_vec
            state = fill_in_state(state)
            state_indicator = fill_in_state(np.ones_like(state_std))
            state_std = fill_in_state(state_std)
            state_mean = fill_in_state(state_mean)
            state_norm = fill_in_state(state_norm)
            # If action's format is different from state's,
            # you may implement fill_in_action()
            actions = fill_in_state(actions)
            
            # Parse the images
            def parse_rgb():
                imgs = []
                for i in range(max(step_id-self.IMG_HISORY_SIZE+1, 0), step_id+1):
                    imgs.append(obs['rgb'][i])
                imgs = np.stack(imgs)
                if imgs.shape[0] < self.IMG_HISORY_SIZE:
                    # Pad the images using the first image
                    imgs = np.concatenate([
                        np.tile(imgs[:1], (self.IMG_HISORY_SIZE-imgs.shape[0], 1, 1, 1)),
                        imgs
                    ], axis=0)
                return imgs
            # `cam_high` is the external camera image
            cam_high = parse_rgb()
            # For step_id = first_idx - 1, the valid_len should be one
            valid_len = min(step_id - start_idx + 1, self.IMG_HISORY_SIZE)
            cam_high_mask = np.array(
                [False] * (self.IMG_HISORY_SIZE - valid_len) + [True] * valid_len
            )
            cam_left_wrist = np.zeros((self.IMG_HISORY_SIZE, 0, 0, 0), dtype=np.uint8)
            cam_left_wrist_mask = np.zeros((self.IMG_HISORY_SIZE,), dtype=bool)
            cam_right_wrist = np.zeros((self.IMG_HISORY_SIZE, 0, 0, 0), dtype=np.uint8)
            cam_right_wrist_mask = np.zeros((self.IMG_HISORY_SIZE,), dtype=bool)
            
            # Return the resulting sample
            # For unavailable images, return zero-shape arrays, i.e., (IMG_HISORY_SIZE, 0, 0, 0)
            # E.g., return np.zeros((self.IMG_HISORY_SIZE, 0, 0, 0)) for the key "cam_left_wrist",
            # if the left-wrist camera is unavailable on your robot
            return True, {
                "meta": meta,
                "state": state,
                "state_std": state_std,
                "state_mean": state_mean,
                "state_norm": state_norm,
                "actions": actions,
                "state_indicator": state_indicator,
                "cam_high": cam_high,
                "cam_high_mask": cam_high_mask,
                "cam_left_wrist": cam_left_wrist,
                "cam_left_wrist_mask": cam_left_wrist_mask,
                "cam_right_wrist": cam_right_wrist,
                "cam_right_wrist_mask": cam_right_wrist_mask
            }

    def parse_hdf5_file_state_only(self, file_path):
        """[Modify] Parse a hdf5 file to generate a state trajectory.

        Args:
            file_path (str): the path to the hdf5 file
        
        Returns:
            valid (bool): whether the episode is valid, which is useful for filtering.
                If False, this episode will be dropped.
            dict: a dictionary containing the training sample,
                {
                    "state": ndarray,           # state[:], (T, STATE_DIM).
                    "action": ndarray,          # action[:], (T, STATE_DIM).
                } or None if the episode is invalid.
        """
        with h5py.File(file_path, 'r') as f:
            qpos = f['observations']['qpos'][:].astype(np.float32)
            actions_raw = f['actions'][:].astype(np.float32)
            num_steps = min(qpos.shape[0], actions_raw.shape[0])
            qpos = qpos[:num_steps]
            actions_raw = actions_raw[:num_steps]
            # [Optional] We drop too-short episode
            if num_steps < self.MIN_EPISODE_LEN:
                return False, None
            
            # [Optional] We skip the first few still steps
            EPS = 1e-2
            # Get the idx of the first qpos whose delta exceeds the threshold
            qpos_delta = np.abs(qpos - qpos[0:1])
            indices = np.where(np.any(qpos_delta > EPS, axis=1))[0]
            if len(indices) > 0:
                first_idx = indices[0]
            else:
                first_idx = 0
            start_idx = max(first_idx - 1, 0)
            target_qpos = actions_raw[:, :7]
            
            # Parse the state and action
            state = qpos[start_idx:]
            action = target_qpos[start_idx:]
            
            # Fill the state/action into the unified vector
            def fill_in_state(values):
                # Target indices corresponding to your state space
                # knife data: 6 arm joints + 1 gripper.
                UNI_STATE_INDICES = [
                    STATE_VEC_IDX_MAPPING[f"arm_joint_{i}_pos"] for i in range(6)
                ] + [
                    STATE_VEC_IDX_MAPPING["gripper_open"]
                ]
                uni_vec = np.zeros(values.shape[:-1] + (self.STATE_DIM,), dtype=np.float32)
                uni_vec[..., UNI_STATE_INDICES] = values
                return uni_vec
            state = fill_in_state(state)
            action = fill_in_state(action)
            
            # Return the resulting sample
            return True, {
                "state": state,
                "action": action
            }

if __name__ == "__main__":
    ds = UR10eHDF5VLADataset()
    for i in range(len(ds)):
        print(f"Processing episode {i}/{len(ds)}...")
        ds.get_item(i)
