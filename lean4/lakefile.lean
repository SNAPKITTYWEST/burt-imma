import Lake
open Lake DSL

package mmep_convergence where
  leanOptions := #[
    ⟨`autoImplicit, false⟩
  ]

require mathlib from git
  "https://github.com/leanprover-community/mathlib4" @ "master"

@[default_target]
lean_lib MMEP_Convergence where
  srcDir := "."

lean_lib MMEP_ProofSketches where
  srcDir := "."

lean_lib BURT_IMMA_Formalization where
  srcDir := "."

lean_lib IMMA where
  srcDir := "."

lean_lib BURT where
  srcDir := "."

lean_lib MetaInvertedSum where
  srcDir := "."

lean_lib BooleanPerceptron where
  srcDir := "."

lean_lib SmoothLeakyActivation where
  srcDir := "."

lean_lib AlexNet_MMEP where
  srcDir := "."

lean_lib SumInversionAgent where
  srcDir := "."

lean_lib SparkDeterministicExecutor where
  srcDir := "."

lean_lib SuperpositionedInduction where
  srcDir := "."

lean_lib AnuQuantumInterference where
  srcDir := "."
