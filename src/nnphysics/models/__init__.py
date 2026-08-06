"""Models layer: the learned predictors and the interface they share.

Importing this module registers every model, so a name in a configuration resolves
without the caller knowing which module implements it.

A model may depend on `data`, on `systems` and on `core`, and on nothing above them. That
is what lets the N-body graph network read the field names that system declares while the
evaluation harness that scores it still cannot see either.
"""

from nnphysics.models.base import (
    CHECKPOINT_SCHEMA_VERSION,
    MODELS,
    Carry,
    ModelContext,
    ModelFactory,
    ModelHyperparameters,
    SurrogateModel,
    build_model,
    load_model,
    model_payload,
    save_model,
)
from nnphysics.models.baselines import (
    CONSTANT_NAME,
    MLP_NAME,
    ConstantModel,
    MultilayerPerceptron,
    build_constant,
    build_mlp,
)
from nnphysics.models.ensemble import ENSEMBLE_NAME, Ensemble
from nnphysics.models.graph import GRAPH_NAME, NBodyGraphModel, build_graph
from nnphysics.models.operator import (
    CONVOLUTION_NAME,
    OPERATOR_NAME,
    ConvolutionalBaseline,
    FourierNeuralOperator,
    build_convolution,
    build_operator,
)

MODELS.add(CONSTANT_NAME, build_constant)
MODELS.add(MLP_NAME, build_mlp)
MODELS.add(GRAPH_NAME, build_graph)
MODELS.add(OPERATOR_NAME, build_operator)
MODELS.add(CONVOLUTION_NAME, build_convolution)

__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "CONSTANT_NAME",
    "CONVOLUTION_NAME",
    "ENSEMBLE_NAME",
    "GRAPH_NAME",
    "MLP_NAME",
    "MODELS",
    "OPERATOR_NAME",
    "Carry",
    "ConstantModel",
    "ConvolutionalBaseline",
    "Ensemble",
    "FourierNeuralOperator",
    "ModelContext",
    "ModelFactory",
    "ModelHyperparameters",
    "MultilayerPerceptron",
    "NBodyGraphModel",
    "SurrogateModel",
    "build_constant",
    "build_convolution",
    "build_graph",
    "build_mlp",
    "build_model",
    "build_operator",
    "load_model",
    "model_payload",
    "save_model",
]
