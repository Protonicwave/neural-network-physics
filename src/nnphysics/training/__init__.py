"""Training layer: one loop, the schedules it follows and the checkpoints it writes.

Nothing here knows which system it is training on. It sees a model through the model
interface and a dataset through the data layer, and that is all a loop needs.
"""

from nnphysics.training.checkpoint import (
    BEST_NAME,
    LAST_NAME,
    CheckpointPaths,
    TrainingState,
    load_checkpoint,
    save_checkpoint,
)
from nnphysics.training.curriculum import Stage, stage_length, stages
from nnphysics.training.history import EpochRecord, TrainingHistory
from nnphysics.training.loop import SHUFFLE_SEED_STREAM, Progress, train_model
from nnphysics.training.losses import (
    HUBER_DELTA,
    LossTerms,
    rollout_error,
    rollout_loss,
    rollout_residual,
)
from nnphysics.training.schedules import cosine_with_warmup, learning_rate_factor

__all__ = [
    "BEST_NAME",
    "HUBER_DELTA",
    "LAST_NAME",
    "SHUFFLE_SEED_STREAM",
    "CheckpointPaths",
    "EpochRecord",
    "LossTerms",
    "Progress",
    "Stage",
    "TrainingHistory",
    "TrainingState",
    "cosine_with_warmup",
    "learning_rate_factor",
    "load_checkpoint",
    "rollout_error",
    "rollout_loss",
    "rollout_residual",
    "save_checkpoint",
    "stage_length",
    "stages",
    "train_model",
]
