import dataclasses
from typing import ClassVar

import einops
import numpy as np

from openpi import transforms


def make_aloha_example() -> dict:
    """Creates a random input example for the Aloha policy."""
    return {
        "state": np.ones((7,)),
        "images": {
            "camera_left_shoulder": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "camera_right_shoulder": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "camera_wrist_left": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        },
        "prompt": "do something",
    }


@dataclasses.dataclass(frozen=True)
class AlohaInputs(transforms.DataTransformFn):
    """Inputs for the Aloha policy.

    Expected inputs:
    - images: dict[name, img] where img is [channel, height, width]. name must be in EXPECTED_CAMERAS.
    - state: [7] - 6 arm joints + 1 gripper (normalized using JOINT normalization by default for Interbotix format)
    - actions: [action_horizon, 7] - same format as state
    
    By default, assumes data was collected with Interbotix version using JOINT normalization.
    Set use_joint_normalization=False for original ALOHA format using POSITION normalization.
    """

    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model.
    adapt_to_pi: bool = True
    
    # If true, assumes data was collected with Interbotix version using JOINT normalization.
    # If false, assumes original ALOHA format using POSITION normalization (linear position).
    # Default is True for Interbotix version (recommended).
    use_joint_normalization: bool = True

    # The expected cameras names. All input cameras must be in this set. Missing cameras will be
    # replaced with black images and the corresponding `image_mask` will be set to False.
    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = (
        "camera_left_shoulder", "camera_right_shoulder", "camera_wrist_left"
    )

    def __call__(self, data: dict) -> dict:
        data = _decode_aloha(data, adapt_to_pi=self.adapt_to_pi, use_joint_normalization=self.use_joint_normalization)

        in_images = data["images"]
        if set(in_images) - set(self.EXPECTED_CAMERAS):
            raise ValueError(f"Expected images to contain {self.EXPECTED_CAMERAS}, got {tuple(in_images)}")

        # Assume that base image always exists.
        base_image = in_images["camera_wrist_left"]

        images = {
            "base_0_rgb": base_image,
        }
        image_masks = {
            "base_0_rgb": np.True_,
        }

        # Add the extra images.
        extra_image_names = {
            "left_shoulder_0_rgb": "camera_left_shoulder",
            "right_shoulder_0_rgb": "camera_right_shoulder",
        }
        for dest, source in extra_image_names.items():
            if source in in_images:
                images[dest] = in_images[source]
                image_masks[dest] = np.True_
            else:
                images[dest] = np.zeros_like(base_image)
                image_masks[dest] = np.False_

        inputs = {
            "image": images,
            "image_mask": image_masks,
            "state": data["state"],
        }

        # Actions are only available during training.
        if "actions" in data:
            actions = np.asarray(data["actions"])
            actions = _encode_actions_inv(actions, adapt_to_pi=self.adapt_to_pi)
            inputs["actions"] = actions

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class AlohaOutputs(transforms.DataTransformFn):
    """Outputs for the Aloha policy.
    
    Converts pi0 model outputs back to Interbotix format (JOINT normalization) by default.
    """

    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model.
    adapt_to_pi: bool = True

    def __call__(self, data: dict) -> dict:
        # Only return the first 7 dims (6 arm joints + 1 gripper for solo).
        actions = np.asarray(data["actions"][:, :7])
        return {"actions": _encode_actions(actions, adapt_to_pi=self.adapt_to_pi)}


def _joint_flip_mask() -> np.ndarray:
    """Used to convert between aloha solo and pi joint angles."""
    return np.array([1, -1, -1, 1, 1, 1, 1])

    # ORIGINAL MASK FOR BIMANUAL
    # return np.array([1, -1, -1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1])


def _normalize(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)


def _unnormalize(x, min_val, max_val):
    return x * (max_val - min_val) + min_val


def _gripper_to_angular(value):
    # Converts normalized gripper position from original ALOHA format (linear position space)
    # to pi0 format (normalized angular space).
    #
    # Input: normalized linear position (0-1) using PUPPET_GRIPPER_POSITION limits (0.01844, 0.05800)
    # Output: normalized angular position (0-1) using pi0 limits (0.5476, 1.6296)
    #
    # NOTE: This function assumes data was collected with original ALOHA format using
    # ROS JointState qpos[7] (linear position). If data was collected with Interbotix
    # version using JOINT normalization, use _gripper_to_angular_from_joint() instead.
    #
    # These values are coming from the Aloha code:
    # PUPPET_GRIPPER_POSITION_OPEN, PUPPET_GRIPPER_POSITION_CLOSED
    value = _unnormalize(value, min_val=0.01844, max_val=0.05800)

    # This is the inverse of the angular to linear transformation inside the Interbotix code.
    def linear_to_radian(linear_position, arm_length, horn_radius):
        value = (horn_radius**2 + linear_position**2 - arm_length**2) / (2 * horn_radius * linear_position)
        return np.arcsin(np.clip(value, -1.0, 1.0))

    # The constants are taken from the Interbotix code.
    value = linear_to_radian(value, arm_length=0.036, horn_radius=0.022)

    # pi0 gripper data is normalized (0, 1) between encoder counts (2405, 3110).
    # There are 4096 total encoder counts and aloha uses a zero of 2048.
    # Converting this to radians means that the normalized inputs are between (0.5476, 1.6296)
    return _normalize(value, min_val=0.5476, max_val=1.6296)


def _gripper_to_angular_from_joint(value):
    # Converts normalized gripper joint angle from Interbotix format to pi0 format.
    #
    # Input: normalized joint angle (0-1) using PUPPET_GRIPPER_JOINT limits (-0.6213, 1.4910)
    # Output: normalized angular position (0-1) using pi0 limits (0.5476, 1.6296)
    #
    # This is the DEFAULT conversion for data collected with Interbotix version that uses
    # JOINT normalization (bot.gripper.get_gripper_position() returns joint angle in radians).
    #
    # These values are coming from the Aloha code:
    # PUPPET_GRIPPER_JOINT_OPEN, PUPPET_GRIPPER_JOINT_CLOSE
    value = _unnormalize(value, min_val=-0.6213, max_val=1.4910)
    
    # pi0 gripper data is normalized (0, 1) between encoder counts (2405, 3110).
    # Converting this to radians means that the normalized inputs are between (0.5476, 1.6296)
    return _normalize(value, min_val=0.5476, max_val=1.6296)


def _gripper_from_angular(value):
    # Convert from the gripper position used by pi0 to the gripper position used by Interbotix format.
    #
    # Input: normalized angular (0-1) using pi0 limits (0.5476, 1.6296)
    # Output: normalized joint angle (0-1) using PUPPET_GRIPPER_JOINT limits (-0.6213, 1.4910)
    #
    # This outputs to Interbotix format (JOINT normalization) which is the standard format
    # for data collected with Interbotix version.

    # We do not scale the output since the trossen model predictions are already in radians.
    # See the comment in _gripper_to_angular for a derivation of the constant
    value = value + 0.5476

    # These values are coming from the Aloha code:
    # PUPPET_GRIPPER_JOINT_OPEN, PUPPET_GRIPPER_JOINT_CLOSE
    return _normalize(value, min_val=-0.6213, max_val=1.4910)


def _gripper_from_angular_inv(value):
    # Directly inverts the gripper_from_angular function.
    # Converts from Interbotix format (normalized joint angle) back to pi0 format.
    #
    # Input: normalized joint angle (0-1) using PUPPET_GRIPPER_JOINT limits (-0.6213, 1.4910)
    # Output: normalized angular (0-1) using pi0 limits (0.5476, 1.6296)
    value = _unnormalize(value, min_val=-0.6213, max_val=1.4910)
    return value - 0.5476


def _decode_aloha(data: dict, *, adapt_to_pi: bool = False, use_joint_normalization: bool = True) -> dict:
    # We Modify from the OG for a single ALOHA arm:
    # OURS:
    # state is [left_arm_joint_angles, left_arm_gripper]
    # dim sizes: [6, 1]
    
    # ORIGINAL :
    # state is [left_arm_joint_angles, left_arm_gripper, right_arm_joint_angles, right_arm_gripper]
    # dim sizes: [6, 1]
    
    state = np.asarray(data["state"])
    state = _decode_state(state, adapt_to_pi=adapt_to_pi, use_joint_normalization=use_joint_normalization)

    def convert_image(img):
        img = np.asarray(img)
        # Convert to uint8 if using float images.
        if np.issubdtype(img.dtype, np.floating):
            img = (255 * img).astype(np.uint8)
        # Convert from [channel, height, width] to [height, width, channel].
        return einops.rearrange(img, "c h w -> h w c")

    images = data["images"]
    images_dict = {name: convert_image(img) for name, img in images.items()}

    data["images"] = images_dict
    data["state"] = state
    return data


def _decode_state(state: np.ndarray, *, adapt_to_pi: bool = False, use_joint_normalization: bool = True) -> np.ndarray:
    if adapt_to_pi:
        # Flip the joints.
        state = _joint_flip_mask() * state
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        if use_joint_normalization:
            # Data collected with Interbotix version using JOINT normalization (default).
            # Input: normalized joint angle (0-1) using PUPPET_GRIPPER_JOINT limits (-0.6213, 1.4910)
            # Output: normalized angular (0-1) using pi0 limits (0.5476, 1.6296)
            state[[6]] = _gripper_to_angular_from_joint(state[[6]])
        else:
            # Original ALOHA format using POSITION normalization (linear position).
            # Input: normalized linear position (0-1) using PUPPET_GRIPPER_POSITION limits (0.01844, 0.05800)
            # Output: normalized angular (0-1) using pi0 limits (0.5476, 1.6296)
            state[[6]] = _gripper_to_angular(state[[6]])

        # ORIGINAL BIMANUAL
        # state[[6]] = _gripper_to_angular(state[[6, 13]])
    return state


def _encode_actions(actions: np.ndarray, *, adapt_to_pi: bool = False) -> np.ndarray:
    if adapt_to_pi:
        # Flip the joints.
        actions = _joint_flip_mask() * actions
        actions[:, [6]] = _gripper_from_angular(actions[:, [6]]) # TODO : typecheck
    return actions 


def _encode_actions_inv(actions: np.ndarray, *, adapt_to_pi: bool = False) -> np.ndarray:
    if adapt_to_pi:
        actions = _joint_flip_mask() * actions
        actions[:, [6]] = _gripper_from_angular_inv(actions[:, [6]]) # TODO : typecheck
    return actions
