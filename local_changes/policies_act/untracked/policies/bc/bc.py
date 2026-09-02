import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch import Tensor
from torch.nn import functional as F

from torchvision.models import resnet18
from policies.common.wrapper import TemporalEnsembling


class BCPolicy(nn.Module):
    def __init__(self, args_override, config=None):
        super().__init__()

        self.action_dim = args_override["action_dim"]
        self.state_dim = args_override["state_dim"]

        self.temporal_ensembler = None

        try:
            if args_override["temporal_agg"]:
                self.temporal_ensembler = TemporalEnsembling(
                    1,
                    self.action_dim,
                    args_override["max_timesteps"],
                )
        except Exception as e:
            print(e)
            print(
                "The above Exception can be ignored when training instead of evaluating."
            )

        # =========================
        # Image Encoder
        # =========================
        backbone = resnet18(pretrained=True)
        backbone.fc = nn.Identity()
        self.backbone = backbone

        image_feature_dim = 512

        # =========================
        # Robot State Encoder
        # =========================
        self.qpos_proj = nn.Sequential(
            nn.Linear(self.state_dim, 128),
            nn.ReLU(),
        )

        # =========================
        # Policy Head
        # =========================
        self.policy_head = nn.Sequential(
            nn.Linear(image_feature_dim + 128, 256),
            nn.ReLU(),
            nn.Linear(256, self.action_dim),
        )

    def reset(self):
        if self.temporal_ensembler is not None:
            self.temporal_ensembler.reset()

    def __call__(self, qpos, image: Tensor, actions=None, is_pad=None):

        if image.ndim == 4:
            image = image.unsqueeze(0)

        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        image = normalize(image)

        B = image.shape[0]

        # =========================
        # Encode image
        # =========================

        # 使用第一个camera
        image = image[:, 0]

        image_feature = self.backbone(image)

        # =========================
        # Encode robot state
        # =========================
        qpos_feature = self.qpos_proj(qpos)

        # =========================
        # Fusion
        # =========================
        feature = torch.cat([image_feature, qpos_feature], dim=-1)

        # =========================
        # Predict action
        # =========================
        pred_action = self.policy_head(feature)

        # =========================
        # Training
        # =========================
        if actions is not None:

            gt_action = actions[:, 0]

            mse = F.mse_loss(pred_action, gt_action)

            return {
                "mse": mse,
                "loss": mse,
            }

        # =========================
        # Inference
        # =========================
        else:

            pred_action = pred_action.unsqueeze(1)

            if self.temporal_ensembler is not None:
                a_hat_one = self.temporal_ensembler.update(pred_action)
                pred_action[0][0] = a_hat_one

            return pred_action

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=1e-4
        )

        return optimizer