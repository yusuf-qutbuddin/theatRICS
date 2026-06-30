## Maximum Entropy Method — what it is and where it comes from

### The general problem MEM solves

In many areas of science you have a measured signal that is the result of a **superposition of many underlying components**, and you want to recover the distribution of those components. The mathematical structure is always an integral equation of this form:

```
G(τ) = ∫ K(τ, x) · f(x) dx
```

Where:
- `G(τ)` is what you measured — the FCS correlation curve
- `K(τ, x)` is a known kernel — the single-species FCS model evaluated at lag time τ for a species with diffusion time x
- `f(x)` is what you want to recover — the distribution of diffusion times

This is called a **Fredholm integral equation of the first kind**. The problem is that inverting it is **ill-posed** — meaning there are infinitely many distributions `f(x)` that are consistent with the measured data `G(τ)` within experimental noise. Small changes in the measured data can lead to wildly different solutions. This is the fundamental mathematical difficulty.

---

### Why naive fitting fails here

If you try to fit a sum of many Gaussians or a large number of discrete diffusion components directly to the data by minimising chi-squared, you get **overfitting**. The optimiser finds a solution that fits the noise perfectly but has wild oscillations — some components get enormous positive amplitudes and others get equally enormous negative amplitudes that cancel out. The result looks like it fits the data but is physically meaningless.

This is not a failure of the optimiser — it is a fundamental property of ill-posed problems. The data simply does not contain enough information to uniquely determine a highly resolved distribution.

---

### What maximum entropy does differently

MEM adds a **regularisation term** to the optimisation that penalises solutions based on how much information they contain. Specifically it maximises:

```
Q = α·S[f] - χ²[f]
```

Where:
- `χ²[f]` is the standard chi-squared — how well the distribution `f` fits the data
- `S[f]` is the **Shannon-Jaynes entropy** of the distribution
- `α` is a regularisation parameter that controls the balance between fit quality and solution simplicity

The entropy of a distribution is:

```
S[f] = -∑ f(x) · ln(f(x)/m(x))
```

Where `m(x)` is a **default model** — your prior belief about what the distribution looks like before seeing the data. Usually this is a flat (uniform) distribution meaning you have no prior preference for any particular diffusion time.

**The entropy term does two things:**

1. It ensures the solution is **non-negative** everywhere — entropy is only defined for positive distributions, so MEM automatically enforces physicality
2. It penalises **sharp features** — a distribution with a sharp peak has lower entropy than a smooth broad one, so MEM prefers the smoothest distribution consistent with the data

The result is the **most featureless distribution that is still consistent with the data**. This is the key philosophical point of MEM — it does not invent structure that is not required by the data, but it faithfully recovers structure that is genuinely present.

---

### The Shannon-Jaynes entropy and its physical meaning

The entropy `S = -∑ p·ln(p)` was originally derived by Shannon for information theory. In the context of MEM, maximising entropy is equivalent to finding the distribution that makes the **fewest assumptions** beyond what the data requires. This is Jaynes' maximum entropy principle from Bayesian probability theory:

> Among all distributions consistent with the constraints (the data), choose the one with maximum entropy — i.e. the one that is maximally non-committal about features the data cannot resolve.

This means:
- If the data is consistent with a single species, MEM will give you a single broad peak
- If the data genuinely requires two species to fit well, MEM will give you two peaks
- MEM will never give you more peaks than the data can support
- The **width** of each peak in the MEM distribution reflects the **uncertainty** in that component's diffusion time, not a physical broadening of the diffusion coefficient

---

### How the algorithm works in practice

The implementation in your current code uses an iterative gradient-based approach. A proper MEM implementation typically uses one of these methods:

**Method 1 — Cambridge algorithm (Skilling and Bryan)**

The gold standard MEM algorithm developed specifically for this type of problem. It works in a subspace of the data space and uses Newton steps to find the saddle point of `Q = αS - χ²`. The key feature is that `α` is determined automatically by the data using a Bayesian evidence criterion.

**Method 2 — Iterative proportional fitting (what your code does)**

Your current implementation does something simpler:
```python
e_G = a_fit_normalized * (alpha_f * D_S - D_r_chi_squared / 2)
a_fit_normalized = a_fit_normalized + e_G * x
```

This is a **steepest ascent** step in the direction of increasing `Q`. The `alpha_f` at each iteration is computed as the ratio of the gradient of chi-squared to the gradient of entropy, scaled by a factor of 20. This is an ad-hoc scaling rather than the proper Bayesian `α` determination.

**Problems with the current implementation:**

1. **`α` is not properly determined** — the ratio `|∂χ²/∂a| / (20·|∂S/∂a|)` is computed element-wise and changes at every iteration. This is not the same as the global regularisation parameter `α` in proper MEM. The factor of 20 is arbitrary.

2. **The normalisation step** `G_fit_normalized = G_fit/G_fit[0] * mean(G[0:10])` is non-standard and introduces a bias — it forces the fit to match the amplitude of the first few data points rather than finding the globally optimal amplitude.

3. **No proper convergence criterion** — the stopping criterion checks whether chi-squared changed by less than `5e-6` over 100 iterations, but this does not guarantee that entropy has been maximised.

4. **No uncertainty estimation** — the distribution `f(x)` is returned as a point estimate with no confidence intervals. Proper MEM (using the Bayesian evidence framework) can provide uncertainty estimates on the recovered distribution.

5. **The default model** is a flat distribution `a_avg = 1/n_tau_D` which is correct in principle, but it is never updated or used in the entropy calculation in the standard Shannon-Jaynes form — the code computes `S = -∑ a·ln(a)` rather than `S = -∑ a·ln(a/m)`.

---

### What a proper MEM implementation should look like for FCS

A correct MEMFCS implementation has these components:

**1. The kernel matrix**

For 3D Gaussian diffusion:

```python
K[i, j] = (1 / ((1 + tau[i]/tau_D[j]) 
           * sqrt(1 + tau[i]/(tau_D[j] * aspect_ratio²))))
```

Shape: `(n_tau_points, n_tau_D_components)`. This encodes how much each diffusion component contributes to the correlation at each lag time.

**2. The forward model**

```
G_fit(τ) = C · K · a
```

Where:
- `a` is the amplitude vector (what we want to find), shape `(n_tau_D,)`
- `C` is a normalisation constant related to N and the PSF
- The amplitude at each τ_D is proportional to the number fraction of molecules with that diffusion time, weighted by their brightness squared

**3. The entropy functional**

```python
S = -np.sum(a * np.log(a / m))   # m is the default model (flat = 1/n_tau_D)
```

**4. The chi-squared functional**

```python
chi2 = np.sum(((G_data - G_fit) / sigma_G) ** 2)
```

**5. The Bryan algorithm for α**

The optimal `α` satisfies:

```
2α · S = n_good
```

Where `n_good` is the number of "good" degrees of freedom — the number of data points that are genuinely constrained by the data. This is determined self-consistently during the optimisation.

**6. Uncertainty estimation**

The posterior distribution of `a` is approximately Gaussian around the MAP estimate, with covariance:

```
Σ = (α·∇²S + ∇²χ²/2)⁻¹
```

The diagonal of `Σ` gives the variance of each amplitude component.

---

### What MEMFCS can and cannot resolve

The resolution of MEMFCS is fundamentally limited by the information content of the data. Specifically:

**Rule of thumb for distinguishability:**

Two species with diffusion times `τ_D1` and `τ_D2` can be resolved if:

```
τ_D2 / τ_D1 > ~3-5 (for typical FCS data quality)
```

This comes from the fact that the FCS correlation function changes slowly with diffusion time — species that differ by less than a factor of ~3 produce nearly identical correlation curves that cannot be distinguished within experimental noise.

**What MEM gives you that standard fitting does not:**

- No assumption about the number of species
- Non-negative amplitudes by construction
- The width of recovered peaks tells you about the reliability of the result — a narrow sharp peak means the data strongly supports that species, a broad smeared peak means the data is consistent with a range of diffusion times
- Multimodal distributions without specifying the number of modes in advance

**What MEM cannot give you:**

- Better resolution than the data contains — if your SNR is low or your τ range is limited, MEM will return broad uninformative peaks regardless of how good the algorithm is
- Absolute concentrations without additional constraints on the amplitude normalisation
- Reliable results if the model (3D Gaussian PSF) does not match the actual experimental PSF

---

### The literature you should be aware of

The key papers for MEMFCS are:

- **Livesey and Brochon (1987)** — first application of MEM to fluorescence lifetime distributions, the conceptual foundation
- **Sengupta, Garai, Balaji, Bhattacharyya, Maiti (2003)** — "Measuring Size Distribution in Highly Heterogeneous Systems with Fluorescence Correlation Spectroscopy" — the key MEMFCS paper for FCS specifically, Biophysical Journal 84:1977-1984. This is where the normalisation convention and the FCS kernel definition come from.
- **Skilling and Bryan (1984)** — the Cambridge MEM algorithm, the computational gold standard
- **Jaynes (1957)** — the philosophical foundation of maximum entropy as a principle of inference

---

### What I would recommend improving in your implementation

Given all of the above, here is what a substantially better MEMFCS would look like:

1. **Proper Shannon-Jaynes entropy** with explicit default model `m`

2. **Automatic α determination** using the Bryan criterion `2αS = n_good` rather than the ad-hoc element-wise ratio

3. **Proper normalisation** — fit the amplitude `C` jointly with the distribution rather than forcing it from the first few data points

4. **Convergence to a proper saddle point** of `Q = αS - χ²/2` rather than just minimising chi-squared

5. **Uncertainty bands** on the recovered distribution — showing not just the MAP estimate but the range of distributions consistent with the data

6. **Cross-validation** — split the data into fitting and validation sets to confirm the recovered distribution generalises

7. **Evidence** — report the Bayesian evidence `ln P(data|model)` so the user can compare MEMFCS against a two-component fit and know which is more supported by the data

Shall I implement a better version of MEMFCS along these lines?