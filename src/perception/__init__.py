"""Perception module exports for SIH 26053 Spatiotemporal 2.5D Perception Engine."""

from .dogma import DogMaBuilder
from .kinematics_engine import KinematicsEngine
from .pipeline import PerceptionPipeline
from .semantic_segmenter import (
    MockSemanticModel,
    SPVCNNSemanticModel,
    SPVNASSemanticModel,
    SemanticModel,
    SemanticSegmenter,
    create_semantic_model,
    map_semantickitti_to_mission,
)
from .spvcnn import SPVCNN, SPVNAS, PointVoxelBlock
from .temporal_stacker import TemporalStacker
from .terrain_analyzer import TerrainAnalyser, TerrainAnalysisResult
from .types import (
    DogmaCell,
    DogMaFrame,
    DynamicObject,
    GlobalDogMa,
    GridScope,
    LocalDogMa,
    ModelArchitecture,
    PerceptionConfig,
    SemanticKITTIClass,
    SemanticLabel,
    SemanticPointCloud,
    WelfordAccumulator,
)
from .validation import validate_labels, validate_point_cloud, validate_timestamp
from .variable_grid import GridCell, QuantisedGrid, VariableGridQuantiser
from .nonuniform_grid import NonUniformGrid, GridPartition, build_default_9partition

from .eigensight_pipeline import (
    EigenSightPipeline,
    PipelineOutput,
    TrackedDynamicObject,
    build_pipeline,
)

__all__ = [
    "PerceptionConfig",
    "SemanticLabel",
    "SemanticKITTIClass",
    "ModelArchitecture",
    "GridScope",
    "SemanticPointCloud",
    "DynamicObject",
    "WelfordAccumulator",
    "DogmaCell",
    "LocalDogMa",
    "GlobalDogMa",
    "DogMaFrame",
    "validate_point_cloud",
    "validate_timestamp",
    "validate_labels",
    "TemporalStacker",
    "SemanticModel",
    "MockSemanticModel",
    "SPVCNNSemanticModel",
    "SPVNASSemanticModel",
    "create_semantic_model",
    "map_semantickitti_to_mission",
    "SPVCNN",
    "SPVNAS",
    "PointVoxelBlock",
    "SemanticSegmenter",
    "KinematicsEngine",
    "GridCell",
    "QuantisedGrid",
    "VariableGridQuantiser",
    "TerrainAnalysisResult",
    "TerrainAnalyser",
    "DogMaBuilder",
    "PerceptionPipeline",
    "EigenSightPipeline",
    "PipelineOutput",
    "TrackedDynamicObject",
    "build_pipeline",
]
