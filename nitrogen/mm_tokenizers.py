import os
from abc import ABC, abstractmethod
from typing import Literal

import numpy as np
import torch
import polars as pl
from pydantic import BaseModel, Field


DEBUG = int(os.getenv("DEBUG", 0))


class Tokenizer(ABC):
    @abstractmethod
    def encode(self, data: dict) -> dict:
        pass

    @abstractmethod
    def decode(self, data: dict) -> dict:
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def eval(self):
        pass


# Token IDs
_PAD_TOKEN = 0
_IMG_TOKEN = 1
_LANG_TOKEN = 2
_PROPRIO_TOKEN = 3
_ACT_TOKEN = 4
_GAME_ID_TOKEN = 6


_UNCONDITIONAL_ID = None


class GameMappingConfig(BaseModel):
    src_files: list[str] = Field(default_factory=list, description="List of source parquet files to build game mapping.")


def get_game_mapping(cfg: GameMappingConfig) -> dict:
    game_set = set()
    for path in cfg.src_files:
        df = pl.read_parquet(path)
        for game in df['game_label'].unique():
            if game == _UNCONDITIONAL_ID:
                continue
            game_set.add(game)
    games = sorted(list(game_set))
    # Set the 0th element to be the unconditional game ID
    games = [_UNCONDITIONAL_ID] + games
    return {game: idx for idx, game in enumerate(games)}


class NitrogenTokenizerConfig(BaseModel):
    tokenizer_id: Literal['nitrogen'] = Field(default='nitrogen', frozen=True)
    training: bool = Field(default=True, description="Whether to apply the transform in training mode.")
    num_visual_tokens_per_frame: int = Field(..., description="Number of visual tokens per frame.")
    max_action_dim: int = Field(..., description="Maximum action dimension.")
    max_sequence_length: int = Field(..., description="Maximum sequence length.")
    action_horizon: int = Field(..., description="Action horizon.")
    game_mapping_cfg: GameMappingConfig | None = Field(default=None, description="Game mapping configuration.")
    old_layout: bool = Field(default=False, description="Something.")


class NitrogenTokenizer(Tokenizer):
    """
    Example transform that prepares video, language, state, and actions
    into a token-based format suitable for your model.

    The sub-methods below (prefixed with `_prepare_`) mirror the original
    modular structure.
    """

    def __init__(self, config: NitrogenTokenizerConfig):
        self.training = config.training
        self.num_visual_tokens_per_frame = config.num_visual_tokens_per_frame
        self.max_action_dim = config.max_action_dim
        self.max_sequence_length = config.max_sequence_length
        self.action_horizon = config.action_horizon
        self.old_layout = config.old_layout

        if config.game_mapping_cfg:
            self.game_mapping = get_game_mapping(config.game_mapping_cfg)
            with open("game_mapping.json", "w") as f:
                import json
                json.dump(self.game_mapping, f, indent=2)
        else:
            self.game_mapping = None

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def _prepare_action(self, data: dict):
        """
        Pad to max_action_dim, return masks.
        """
        if "action" not in data:
            actions = np.zeros((self.action_horizon, self.max_action_dim))
            actions_mask = np.zeros((self.action_horizon, self.max_action_dim), dtype=bool)
            n_action_tokens = self.action_horizon
            return actions, actions_mask, n_action_tokens

        actions = data["action"]
        assert actions.shape[0] == self.action_horizon, \
            f"{actions.shape=}, {self.action_horizon=}"

        n_action_tokens = actions.shape[0]
        n_action_dims = actions.shape[1]

        assert n_action_dims <= self.max_action_dim, \
            f"Action dim {n_action_dims} exceeds max {self.max_action_dim}."

        actions = np.pad(actions, ((0, 0), (0, self.max_action_dim - n_action_dims)), "constant")

        actions_mask = np.zeros((n_action_tokens, self.max_action_dim), dtype=bool)
        actions_mask[:, :n_action_dims] = True

        return actions, actions_mask, n_action_tokens

    def _build_token_ids(self, n_images: int, n_action_tokens: int):
        """
        Build the 1D array of token_ids based on the number of each block.
        Return (token_ids, special_pad_token_idx).
        """
        vl_token_ids = []
        sa_token_ids = []

        # 0.5) Add a Game ID placeholder
        if self.game_mapping:
            vl_token_ids.append(_GAME_ID_TOKEN)

        # 1) Video placeholders
        for _ in range(n_images):
            vl_token_ids.extend([_IMG_TOKEN] * self.num_visual_tokens_per_frame)

        # 2) Action tokens
        sa_token_ids.extend([_ACT_TOKEN] * n_action_tokens)

        return np.array(vl_token_ids), np.array(sa_token_ids)

    def _prepare_attention_mask(self, vl_token_ids: np.ndarray):
        """
        Build 1D attention mask for vision-language tokens.
        1 indicates valid token, 0 indicates padding token.
        State-action attention will be handled separately by the model.
        """
        # Only create attention mask for vision-language tokens
        vl_seq_len = vl_token_ids.shape[0]
        vl_attn_mask = np.ones(vl_seq_len, dtype=bool)  # All tokens are valid initially

        if vl_seq_len > self.max_sequence_length:
            raise ValueError("VL sequence length exceeds the max sequence length!")

        left_pad_len = self.max_sequence_length - vl_seq_len

        # Pad token_ids (with PAD_TOKEN)
        vl_token_ids = np.pad(vl_token_ids, (left_pad_len, 0), constant_values=_PAD_TOKEN)

        # Pad attention mask with 0 (padding tokens)
        vl_attn_mask = np.pad(vl_attn_mask, (left_pad_len, 0), constant_values=0)

        return vl_token_ids, vl_attn_mask

    def pack_actions(self, buttons, j_left, j_right):
        """Pack buttons and joysticks into single action tensor."""
        # Check that the first two dims of each input is the same (number of chunks, control frequency)
        assert buttons.shape[:2] == j_left.shape[:2] == j_right.shape[:2]

        # Normalize the joysticks to 0,1
        j_left = (j_left + 1) / 2.0
        j_right = (j_right + 1) / 2.0

        # Concatenate the buttons and joysticks along the last dimension
        action = np.concatenate([buttons, j_left, j_right], axis=-1, dtype=np.float32)
        # Squeeze the first dimension of each input: this is the number of chunks, which is 1 here
        action = action.squeeze(0)
        return action

    def unpack_actions(self, actions):
        """Unpack action tensor back to joysticks and buttons."""
        if self.old_layout:
            # Unpack the actions into j_left, j_right, buttons
            j_left = actions[:, :, :2]
            j_right = actions[:, :, 2:4]
            buttons = actions[:, :, 4:]
        else:
            # Unpack the actions into j_left, j_right, buttons
            buttons = actions[:, :, :-4]
            j_left = actions[:, :, -4:-2]
            j_right = actions[:, :, -2:]

        # Denormalize the joysticks back to -1,1
        j_left = j_left * 2.0 - 1.0
        j_right = j_right * 2.0 - 1.0

        # Clip into [-1,1]
        j_left = torch.clamp(j_left, -1, 1)
        j_right = torch.clamp(j_right, -1, 1)
        # Threshold the buttons to 0/1
        buttons = (buttons > 0.5).float()

        return j_left, j_right, buttons

    def encode(self, data: dict) -> dict:
        """
        Main entry point for the transform. We assume that `data` has
        data['video'], data['language'], data['state'], and data['action'] in
        the shapes needed. If you have multiple keys for each modality, you
        could use your own grouping logic (similar to GR1Transform) first.
        """

        # 1) Pack buttons/joysticks into a single action tensor


        transformed_data = {**data}  # Start with a copy of the input data

        n_images = (data["dropped_frames"] == False).sum()
        transformed_data["images"] = data["frames"]
        transformed_data["dropped_images"] = data["dropped_frames"]

        if self.training:
            # Keep the original actions in the data for evaluation
            packed_actions = self.pack_actions(
                data["buttons"],
                data["j_left"],
                data["j_right"]
            )
            data["action"] = packed_actions

            transformed_data["has_real_action"] = np.ones((), dtype=bool)

            actions, actions_mask, n_action_tokens = self._prepare_action(data)
            transformed_data["actions"] = actions
            transformed_data["actions_mask"] = actions_mask
        else:
            n_action_tokens = self.action_horizon

        transformed_data["has_detection_target"] = np.zeros((), dtype=bool)

        # 5) Build token_ids
        vl_token_ids, sa_token_ids = self._build_token_ids(
            n_images, n_action_tokens
        )

        # 6) Build the attention mask only for vision-language tokens
        vl_token_ids, vl_attn_mask = self._prepare_attention_mask(vl_token_ids)

        transformed_data["vl_token_ids"] = vl_token_ids
        transformed_data["sa_token_ids"] = sa_token_ids
        transformed_data["vl_attn_mask"] = vl_attn_mask
        transformed_data["embodiment_id"] = torch.tensor(0, dtype=torch.long)

        if self.game_mapping:
            game_name = data["game"]
            assert game_name in self.game_mapping, f"Game '{game_name}' not found in game mapping."
            transformed_data["game_ids"] = torch.tensor(self.game_mapping[game_name], dtype=torch.long)
        else:
            transformed_data["game_ids"] = torch.tensor(0, dtype=torch.long)
        return transformed_data

    def decode(self, data: dict) -> dict:
        j_left, j_right, buttons = self.unpack_actions(data["action_tensor"])
        return {
            "j_left": j_left,
            "j_right": j_right,
            "buttons": buttons,
        }