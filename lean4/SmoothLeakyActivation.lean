/-
  SmoothLeakyActivation
  Smooth leaky activation function with bounded gradient properties
-/
import Mathlib
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Analysis.Calculus.Deriv.Basic

noncomputable section

open Real

-- ============================================================
-- Parameters
-- ============================================================

structure Params where
  alpha : ℝ
  beta : ℝ
  h_alpha_pos : 0 < alpha
  h_alpha_lt_one : alpha < 1
  h_beta_pos : 0 < beta

-- ============================================================
-- Activation Function
-- Standard smooth leaky ReLU: f(x) = (1+α)/2 * x + (1-α)/2 * (1/β) * log(cosh(βx))
-- Derivative: f'(x) = (1+α)/2 + (1-α)/2 * tanh(βx) ∈ (α, 1)
-- ============================================================

/-- The smooth leaky activation function -/
def f (p : Params) (x : ℝ) : ℝ :=
  (1 + p.alpha) / 2 * x + (1 - p.alpha) / 2 * (1 / p.beta) * Real.log (Real.cosh (p.beta * x))

/-- The derivative of the smooth leaky activation -/
def f' (p : Params) (x : ℝ) : ℝ :=
  (1 + p.alpha) / 2 + (1 - p.alpha) / 2 * Real.tanh (p.beta * x)

-- ============================================================
-- Properties
-- ============================================================

/-- The gradient is always strictly positive -/
theorem deriv_pos (p : Params) : ∀ x : ℝ, f' p x > 0 := by
  intro x
  have h₁ : f' p x = (1 + p.alpha) / 2 + (1 - p.alpha) / 2 * Real.tanh (p.beta * x) := rfl
  rw [h₁]
  have h₂ : Real.tanh (p.beta * x) > -1 := Real.neg_one_lt_tanh (p.beta * x)
  have h₄ : 0 < p.alpha := p.h_alpha_pos
  have h₅ : p.alpha < 1 := p.h_alpha_lt_one
  have h₇ : 0 < 1 - p.alpha := by linarith
  have h₈ : 0 < 1 + p.alpha := by linarith
  have h₁₀ : (1 + p.alpha) / 2 + (1 - p.alpha) / 2 * Real.tanh (p.beta * x) ≥
      (1 + p.alpha) / 2 + (1 - p.alpha) / 2 * (-1) := by
    have h₁₁ : Real.tanh (p.beta * x) ≥ -1 := by linarith
    nlinarith
  have h₁₁ : (1 + p.alpha) / 2 + (1 - p.alpha) / 2 * (-1 : ℝ) = p.alpha := by ring
  linarith [p.h_alpha_pos]

/-- The gradient is bounded below 1 -/
theorem deriv_bounded (p : Params) : ∀ x : ℝ, f' p x < 1 := by
  intro x
  have h₁ : f' p x = (1 + p.alpha) / 2 + (1 - p.alpha) / 2 * Real.tanh (p.beta * x) := rfl
  rw [h₁]
  have h₂ : Real.tanh (p.beta * x) < 1 := Real.tanh_lt_one (p.beta * x)
  have h₅ : p.alpha < 1 := p.h_alpha_lt_one
  have h₇ : 0 < 1 - p.alpha := by linarith
  have h₁₀ : (1 + p.alpha) / 2 + (1 - p.alpha) / 2 * Real.tanh (p.beta * x) <
      (1 + p.alpha) / 2 + (1 - p.alpha) / 2 * (1 : ℝ) := by nlinarith
  have h₁₁ : (1 + p.alpha) / 2 + (1 - p.alpha) / 2 * (1 : ℝ) = 1 := by ring
  linarith

/-- As x → +∞, f'(x) → 1 -/
theorem deriv_limit_pos_inf (p : Params) :
    Filter.Tendsto (f' p) Filter.atTop (nhds 1) := by
  have htanh : Filter.Tendsto (fun x : ℝ => Real.tanh (p.beta * x)) Filter.atTop (nhds 1) := by
    have hbeta : Filter.Tendsto (fun x : ℝ => p.beta * x) Filter.atTop Filter.atTop :=
      Filter.Tendsto.atTop_mul_atTop (by linarith [p.h_beta_pos]) Filter.tendsto_id
    exact Real.tendsto_tanh_atTop.comp hbeta
  have : Filter.Tendsto (fun x : ℝ => (1 + p.alpha) / 2 + (1 - p.alpha) / 2 * Real.tanh (p.beta * x))
      Filter.atTop (nhds ((1 + p.alpha) / 2 + (1 - p.alpha) / 2 * 1)) :=
    Filter.Tendsto.const_add _ (Filter.Tendsto.const_mul _ htanh)
  have heq : (1 + p.alpha) / 2 + (1 - p.alpha) / 2 * 1 = 1 := by ring
  simp only [f']
  rwa [heq] at this

/-- As x → -∞, f'(x) → alpha -/
theorem deriv_limit_neg_inf (p : Params) :
    Filter.Tendsto (f' p) Filter.atBot (nhds p.alpha) := by
  have htanh : Filter.Tendsto (fun x : ℝ => Real.tanh (p.beta * x)) Filter.atBot (nhds (-1)) := by
    have hbeta : Filter.Tendsto (fun x : ℝ => p.beta * x) Filter.atBot Filter.atBot :=
      Filter.Tendsto.atBot_mul_atBot (by linarith [p.h_beta_pos]) Filter.tendsto_id
    exact Real.tendsto_tanh_atBot.comp hbeta
  have : Filter.Tendsto (fun x : ℝ => (1 + p.alpha) / 2 + (1 - p.alpha) / 2 * Real.tanh (p.beta * x))
      Filter.atBot (nhds ((1 + p.alpha) / 2 + (1 - p.alpha) / 2 * (-1))) :=
    Filter.Tendsto.const_add _ (Filter.Tendsto.const_mul _ htanh)
  have heq : (1 + p.alpha) / 2 + (1 - p.alpha) / 2 * (-1) = p.alpha := by ring
  simp only [f']
  rwa [heq] at this

/-- f is C^∞ (infinitely differentiable) -/
theorem smooth (p : Params) : ContDiff ℝ ⊤ (f p) := by
  have h_lin : ContDiff ℝ ⊤ (fun x : ℝ => (1 + p.alpha) / 2 * x) :=
    (contDiff_const.mul contDiff_id)
  have h_cosh : ContDiff ℝ ⊤ (fun x : ℝ => Real.cosh (p.beta * x)) :=
    Real.contDiff_cosh.comp (contDiff_const.mul contDiff_id)
  have h_pos : ∀ x : ℝ, Real.cosh (p.beta * x) > 0 := fun x => Real.cosh_pos _
  have h_log : ContDiff ℝ ⊤ (fun x : ℝ => Real.log (Real.cosh (p.beta * x))) :=
    h_cosh.log (fun x => ne_of_gt (h_pos x))
  have h_scaled : ContDiff ℝ ⊤ (fun x : ℝ => (1 - p.alpha) / 2 * (1 / p.beta) * Real.log (Real.cosh (p.beta * x))) :=
    contDiff_const.mul h_log
  simp only [f]
  exact h_lin.add h_scaled

/-- f(x) < 0 for all x < 0 -/
theorem neg_range (p : Params) : ∀ x : ℝ, x < 0 → f p x < 0 := by
  intro x hx
  simp only [f]
  have h₄ : 0 < p.beta := p.h_beta_pos
  have h₅ : 0 < 1 + p.alpha := by linarith [p.h_alpha_pos]
  have h₆ : 0 < 1 - p.alpha := by linarith [p.h_alpha_lt_one]
  have h₇ : (0 : ℝ) < 1 / p.beta := by positivity
  have h₈ : Real.log (Real.cosh (p.beta * x)) ≥ 0 :=
    Real.log_nonneg (Real.one_le_cosh _)
  have h₉ : (1 + p.alpha) / 2 * x < 0 := by nlinarith
  have h₁₂ : Real.log (Real.cosh (p.beta * x)) ≤ p.beta * |x| := by
    have hle : Real.cosh (p.beta * x) ≤ Real.exp (p.beta * |x|) := by
      calc Real.cosh (p.beta * x)
          ≤ Real.cosh (|p.beta * x|) := Real.cosh_abs _ ▸ le_refl _
        _ = Real.cosh (p.beta * |x|) := by
            rw [abs_mul, abs_of_pos h₄]
        _ ≤ Real.exp (p.beta * |x|) := Real.cosh_le_exp _
    linarith [Real.log_le_log (Real.cosh_pos _) hle,
              Real.log_exp (p.beta * |x|)]
  have h₁₃ : |x| = -x := abs_of_neg hx
  rw [h₁₃] at h₁₂
  have h₁₄ : (1 - p.alpha) / 2 * (1 / p.beta) * Real.log (Real.cosh (p.beta * x))
      ≤ (1 - p.alpha) / 2 * (-x) := by
    have : (1 - p.alpha) / 2 * (1 / p.beta) * (p.beta * (-x)) = (1 - p.alpha) / 2 * (-x) := by
      field_simp
    nlinarith
  have h₁₅ : (1 + p.alpha) / 2 * x + (1 - p.alpha) / 2 * (-x) = p.alpha * x := by ring
  linarith [mul_neg_of_pos_of_neg p.h_alpha_pos hx]

/-- f(0) = 0 -/
theorem f_zero (p : Params) : f p 0 = 0 := by
  simp [f, Real.cosh_zero, Real.log_one]

end
