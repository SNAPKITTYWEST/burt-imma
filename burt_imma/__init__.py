"""
BURT-IMMA: BiEncoder Unified Retrieval-Transformer with
Instruction, Memory, and Mixture of Experts Agents

Matrix-Memory Equilibrium Propagation (MMEP) implementation.

Components:
  - BURT: Bidirectional Universal Retrieval Transformer (retrieval phase)
  - IMMA: Instruct-MoE Matrix-Memory Architecture (generation phase)
  - MMEP: Matrix-Memory Equilibrium Propagation (training algorithm)

Contact: jessica@collectivekitty.com
License: BSL-1.1
"""

__version__ = "0.1.0"
__author__ = "SnapKitty West / Bel Esprit D'Accord Irrevocable Trust"

from .burt import BURTRetriever, constrained_softmax
from .imma import IMMAExpert, IMMALayer
from .perceptron_actor import PerceptronActor, PerceptronNetwork, Signal, SignalType
from .meta_inverted_sum import meta_inverted_sum_softmax, huntington_softmax
from .perceptron_experts import PerceptronExpert, PerceptronRouter
from .smooth_leaky import SmoothLeakyActivation, SmoothLeakyMMEP
from .sum_inversion import IntegratedSumInversionAgent, IntegratedAgentConfig
from .superpositioned_induction import (
    SuperpositionedInductionHeads, IterativeSuperpositionRefinement,
    SuperpositionedBURTIMMA
)
from .quantum_interference import QuantumIntegratedSystem, QuantumInterferenceResolver
from .spark_executor import DeterministicExecutor, PerceptronDispatchEngine
from .alexnet_mmep import AlexNetMMEP, EntropyConstrainedRouter
from .unified_architecture import UnifiedDeterministicSystem
