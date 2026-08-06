"""The graph network for the N-body system."""

from nnphysics.models.graph.model import GRAPH_NAME, NBodyGraphModel, build_graph
from nnphysics.models.graph.network import InteractionNetwork

__all__ = ["GRAPH_NAME", "InteractionNetwork", "NBodyGraphModel", "build_graph"]
