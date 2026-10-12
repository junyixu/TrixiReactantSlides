#import "@preview/touying:0.6.1": *
#import themes.metropolis: *

#show: metropolis-theme.with(
  aspect-ratio: "16-9",
  footer: self => self.info.title,
  config-info(
    title: [Trixi.jl on TPUs via Reactant.jl],
    subtitle: [Feasibility study — progress report],
    author: [Junyi Xu],
    date: datetime(year: 2026, month: 10, day: 12),
    institution: [Summer research],
  ),
)

#set text(font: "Adwaita Sans", size: 19pt)
#show raw: set text(font: "Fira Code", size: 0.9em, features: (calt: 0))  // no `===` ligature
#set table(inset: (x: 6pt, y: 4pt), stroke: 0.5pt + luma(180))
#show table: set text(size: 15pt)

#let ok = text(fill: rgb("#2e7d32"), weight: "bold")[✓]
#let no = text(fill: rgb("#c62828"), weight: "bold")[✗]
#let warn = text(fill: rgb("#ef6c00"), weight: "bold")[!]

#title-slide()

= Outline <touying:hidden>

#outline(title: none, indent: 1em, depth: 1)

= Question and background

== The question

*Can Trixi.jl run on TPUs through Reactant.jl, and how far does "run" go?*

- Reactant traces Julia code into MLIR / StableHLO and compiles it with XLA for CPU, GPU and TPU.
- GPU kernels written with CUDA.jl or KernelAbstractions.jl (KA) can be *raised*: the
  scalar per-thread kernel is rewritten as tensor operations in StableHLO.
- On non-NVIDIA targets (TPU, Trainium) raising is mandatory and switched on automatically
  (`src/compiler/Compiler.jl` in Reactant).
- So "Trixi on TPU" means: every kernel that Trixi launches must be raisable, and every data
  structure the kernels touch must be an XLA buffer of plain numbers.

== Two existing GPU lines in the Trixi ecosystem

#table(
  columns: (auto, 1fr, 1fr),
  [], [*TrixiCUDA.jl*], [*Upstream Trixi.jl GPU path*],
  [Kernels], [hand-written CUDA.jl kernels re-implementing `rhs!`], [KA kernels inside Trixi (issue \#2822)],
  [Mesh], [`TreeMesh` 1D/2D/3D], [`P4estMesh` 2D/3D (`TreeMesh` "not planned")],
  [On the GPU], [semidiscretization only; time stepping = OrdinaryDiffEq on `CuArray`; some callbacks copy to host], [rhs, stepsize and analysis callbacks; `storage_type`, `real_type` options],
  [Trixi version], [pinned to 0.11 – 0.13.5], [current (0.17)],
)

Both are reachable from Reactant in principle (it intercepts `@cuda` and KA launches), but the
upstream KA path is the one that can be contributed back.

== What "running on a TPU" could mean

#table(
  columns: (1fr, auto, 1.3fr),
  [*Level*], [*Status*], [*Evidence / reason*],
  [`rhs!` from hand-written kernels], [#ok], [raised kernels run on a real TPU, E2],
  [`rhs!` from upstream Trixi's KA kernels], [#no], [4 of 20 kernels raise in every case; volume terms do not (E4)],
  [whole explicit time loop, fixed `dt`, in *one* XLA program], [#ok], [`@trace for` over 5-stage RK, E2],
  [CFL-adaptive `dt`], [#warn], [should work (`@trace while`, in-graph reduction); not tested],
  [callbacks with I/O], [host], [compile "advance k steps", do I/O between calls],
  [AMR, p4est mesh generation], [#no], [XLA needs static shapes; p4est is a C library on the host],
)

TrixiCUDA currently stops at the `rhs!` level; with Reactant the time loop itself can be compiled.

= Experiments

== Setup

- RunPod RTX 3070 (8 GB, E1–E3) and RTX A4500 (20 GB, E4), both Ampere.
- Julia 1.12, Reactant 0.2.291, CUDA.jl 6.2, Trixi 0.17.14 with
  `backend = "kernelabstractions"`, `loop_vectorization = false`.
- *TPU proxy*: Reactant's CPU backend with `raise = true` — the kernel must be raised, exactly
  as on a TPU.
- *Real TPU*: Cloud TPU v5e, 1 chip (`v5litepod-1`, europe-west4-b; spot €0.49/h, on-demand
  €1.37/h when spot was preempted), running the `:tpu` variant of the image.
- Scripts and notes: private repo `junyixu/TrixiReactant.jl`; reproducible GPU image
  `ghcr.io/junyixu/trixidocker:latest` / `:tpu` (built by GitHub Actions).

== E1: device check and FP32 vs FP64

#table(
  columns: (auto, auto, auto, auto),
  [*Backend*], [*Type*], [*axpy, $n = 2^24$*], [*sin/cos reduction*],
  [GPU], [Float32], [0.66 ms (305 GB/s)], [0.31 ms],
  [GPU], [Float64], [1.22 ms (331 GB/s)], [5.17 ms],
  [CPU], [Float32], [5.13 ms], [1.16 ms],
  [CPU], [Float64], [9.39 ms], [3.48 ms],
)

- Reactant drives the 3070 at ≈70 % of peak bandwidth with no CUDA.jl code.
- Compute-bound work is *≈17× slower in Float64* on this consumer GPU.

== E2: TrixiCUDA-style kernels through Reactant

1D DGSEM, Burgers, LLF flux, periodic, $N = 3$, 5-stage low-storage RK, 100 steps. Kernels
written in TrixiCUDA style (`@cuda`, `threadIdx`, neighbour index arrays).

#table(
  columns: (1.6fr, auto, auto, auto, auto),
  [*Mode*], [*16k el. F32*], [*16k F64*], [*1M el. F32*], [*CPU backend*],
  [native CUDA.jl], [17.2 ms], [17.2 ms], [*256 ms*], [—],
  [Reactant, kernels kept as PTX], [9.9 ms], [13.7 ms], [252 ms], [7006 ms],
  [Reactant, *raised* to StableHLO], [*3.1 ms*], [12.5 ms], [413 ms], [*79 ms*],
  [raised, shared memory + `sync_threads`], [#no], [#no], [#no], [#no],
)

All working modes reproduce the same-precision CPU reference *bit for bit* after 100 steps.

== E2: what we learned

- *Whole-loop compilation pays off for small problems*: one XLA program instead of a host loop
  launching ≈15 kernels per step → 5.5× faster than native CUDA.jl at 65k DOFs.
- *Raised code is not free*: at 4M DOFs it is 1.6× slower than the hand-written kernels.
- *Shared-memory kernels do not raise*: `failed to raise func` on GPU and CPU backends.
  TrixiCUDA's optimized kernels and Trixi's 3D LDS volume kernels use this pattern.
- Without raising, the CPU backend still runs the CUDA kernel (lowered to CPU code), but
  ≈90× slower than the raised version — and a TPU has no such fallback.

== E3: upstream Trixi through Reactant — blockers

Target: `p4est_2d_dgsem/elixir_advection_basic.jl`, `Trixi.rhs_hyperbolic!` under `@compile`. Not reached yet.

#table(
  columns: (1.2fr, 1fr),
  [*Blocker*], [*Workaround (method overrides only)*],
  [`wrap_array` uses `unsafe_wrap(pointer(u_ode))`], [return a `ReshapedArray` (Reactant writes back)],
  [containers build views with `unsafe_wrap_or_alloc` (pointers)], [reshaped device copy (no AMR)],
  [boundary/interface data stored as arrays of `Symbol` tuples], [keep on host / encode as `Int8`],
  [kernels branch on `node_indices` with `===` on symbols], [integer index type + methods for the 5 index helpers],
)

On CUDA the last point works only because a `CuArray` stores *pointers* to interned symbols;
an XLA buffer cannot. No Trixi source was changed — all workarounds are added methods.

== E4: inventory of Trixi's 20 KA kernels — method

- 16 cases with the elixirs and kernel variants of `test_cuda_2d/3d.jl`: advection, Euler
  (weak form, flux differencing with every sweep variant), MHD; 2D and 3D; Float32.
- A *recording KA backend* captures every launch of one `compute_coefficients` and one `rhs!`
  on the CPU, with KA's CPU backend as reference (native CUDA where that fails).
- Each kernel is *replayed alone* under `@compile raise = ...`; only its arguments are
  converted, which sidesteps the E3 blockers.
- Modes: through Reactant on the GPU *without* raising ("kept"), raised on the GPU, raised on
  the CPU (TPU proxy). One process per case; a crashing kernel is skipped and the case rerun.
- No Trixi source changed; harness workarounds on the next slides.

== E4: results

#table(
  columns: (1.7fr, auto, auto, 1.7fr),
  [*Kernel (definitions)*], [*kept*], [*raised*], [*Blocker when raised*],
  [`prolong2boundaries` (2D, 3D)], [#ok], [#ok], [—],
  [`prolong2interfaces_and_calc_interface_flux` (2D, 3D)], [#ok], [#ok #warn], [2D advection: 80 of 4096 outputs NaN],
  [`calc_boundary_flux` (2D, 3D)], [#ok], [3D #ok, 2D #no], [`sinpi` in the boundary data (`__nv_sinpif`)],
  [`calc_surface_integral_…_sources` (2D, 3D)], [#ok], [#warn], [raised only in 2D without source terms; stack `alloca`, `scf.while`],
  [`volume_integral` (weak form; 2D flux differencing)], [#ok], [#no], [`scf.while` loop; 3D advection: compiler segfault],
  [`flux_differencing`, `…_turbo` (10 methods, 3D)], [#ok], [#no], [shared memory + `@synchronize`; FullSweepGlobal, turbo FullSweep: segfault],
  [`compute_coefficients` (initial condition)], [#ok], [#no], [`scf.while` loop, `sinpi`],
)

*Kept*: all 20 match the reference (rel. diff. ≤ $4 times 10^(-7)$, many bitwise).
GPU and CPU raising agree everywhere.

== E4: what this means

- *4 of 20 kernel definitions raise in every case, 2 in some, 14 never* — the 14 include
  every volume-term kernel, i.e. most of the arithmetic.
- The surface kernels that do raise contain *20–42 gathers and up to 36 scatters* each, the
  pattern that is slow on a TPU (E5).
- Most blockers are compiler gaps, not Trixi design: loops the raiser does not recognise
  (`scf.while`), small stack arrays, a missing `sinpi` lowering, a segfault, a miscompile.
- Shared memory: Reactant does not lower barriers yet (its own test suite: "We don't yet auto
  lower this via polygeist"). On a TPU it is pointless anyway — XLA manages on-chip memory —
  so the global-memory variants are the ones to target.

== E4: bugs and gaps found (to report)

#table(
  columns: (auto, 1fr, auto),
  [*Where*], [*Issue*], [*Workaround in E4*],
  [Reactant], [kernel arguments of type `Type` rejected as non-isbits; every Trixi kernel receives `typeof(mesh)` (CUDA.jl passes it as a zero-size ghost)], [ghost via two methods],
  [Reactant], [raised 2D interface kernel returns NaN in 80/4096 outputs (bitwise correct without raising)], [—],
  [Reactant], [segfault in `AffineCFGPass` (`OptimizeRem`) while raising 3 kernels], [skip kernel],
  [Reactant], [not raised: `scf.while` loops, stack `alloca`, `__nv_sinpif`, barriers], [—],
  [Trixi], [3D HalfSweep kernels fail on KA's *CPU* backend: locals across `@synchronize` without `@private`], [CUDA reference],
  [Trixi], [basis/mortar types require `<: AbstractMatrix{RealT}`; traced arrays have eltype `TracedRNumber{RealT}`], [wrapper type],
  [Trixi], [`node_indices` stored as `Symbol` tuples, compared with `===` in kernels], [`Int8` codes],
)

== First run on a real TPU (v5e, 1 chip)

#table(
  columns: (1.5fr, auto, auto, auto),
  [*E2, 100 RK steps*], [*16k el. F32*], [*16k el. F64*], [*1M el. F32*],
  [TPU v5e, raised], [143.7 ms], [295.7 ms (emulated)], [10.5 s],
  [RTX 3070, raised], [3.1 ms], [12.5 ms], [413 ms],
  [RTX 3070, native CUDA.jl], [17.2 ms], [17.2 ms], [256 ms],
)

- The *CUDA.jl kernels run on the TPU after raising*; Float32 results match the CPU bit for bit.
- Shared-memory kernel: raising fails, as on the other backends.
- No `dot_general` in this program, so `dot_general_precision` makes no difference here.
- *But it is slow*: 25× slower than the 3070 at 4M DOFs. Cause: the scalar gathers of the
  raised kernels (E5, next slide).
- E1 on the TPU: Float32 axpy at 426 GB/s (peak 819); Float64 runs but is emulated —
  sin/cos reduction 170× slower than Float32.

== E5: why raised kernels are slow on the TPU

E2's 1D DG solver on the v5e, 100 RK steps, everything compiled by Reactant:

#table(
  columns: (1.8fr, auto, auto, auto),
  [*Variant*], [*16k el.*], [*1M el.*], [*gathers*],
  [raised kernels, neighbours from index arrays (= E2)], [143.9 ms], [10.5 s], [4],
  [raised kernels, neighbours computed], [58.5 ms], [3.75 s], [2],
  [raised kernels, element-major storage], [145.0 ms], [10.6 s], [4],
  [*array formulation* (matrix product + slices)], [*3.3 ms*], [*186 ms*], [0],
  [RTX 3070: native CUDA.jl / raised (E2)], [17.2 / 3.1 ms], [256 / 413 ms], [—],
)

- *Not the layout*: XLA already stores the arrays element-major on the TPU (`{0,1:T(4,128)}`).
- *The gathers*: raised kernels gather one scalar per element; run time follows their number.
- *As array operations the same solver is 57× faster* — and beats the 3070's native CUDA at
  4M DOFs. Results: kernels bitwise equal to the CPU; array form $4 times 10^(-6)$.

= Precision

== Do we need Float64?

- *TPUs have no native Float64*: on the v5e, Float64 runs emulated — 2× slower in E2, 170× in
  a compute-bound reduction (E1), and no longer bitwise reproducible ($8 times 10^(-14)$).
- *Consumer GPUs*: FP64 at 1/64 of FP32 (17× slower measured on the 3070).
- *Upstream Trixi already supports Float32*: `semidiscretize(...; real_type = Float32)`;
  the CUDA/AMDGPU tests run Float32 with Float32 tolerances.
- *TPU pitfall*: XLA computes Float32 matrix products in bf16 by default. Raised DG kernels
  are likely to contain `dot_general`, so Reactant must be told to use full precision:
  `Reactant.with_config(; dot_general_precision = PrecisionConfig.HIGHEST)`.

== Float32: what to check before relying on it

- In E2 the Float32 *rhs* error grows with resolution (relative $10^(-2)$ at 16k elements,
  $0.58$ at 1M) because $dif u slash dif t = J^(-1) (dots)$ is a difference of nearly equal terms — yet the solution after 100 steps
  is still accurate to $10^(-5)$.
- Planned check: experimental order of convergence in Float32 vs Float64 on standard
  convergence tests, plus long-time runs, entropy-conservative fluxes (log means) and
  steady-state problems.
- *Question for you*: are the target applications fine with Float32?

= Next steps

== Plan

+ *Report the Reactant issues* with minimal reproducers: `Type` kernel arguments, the NaN
  miscompile, the `AffineCFGPass` segfault, unraised loops (`scf.while`) and `sinpi`.
+ *Array formulation for the TPU*: prototype the P4est 2D `rhs!` as array operations (volume
  term as one `dot_general`, interfaces via precomputed index maps on whole faces) in a
  `TrixiReactantExt` package extension, then measure on the v5e against E5's 1D numbers.
+ Report the Trixi issues (`@private` for KA's CPU backend, basis type constraints, symbolic
  `node_indices`) on \#2822; cover mortars / non-conforming meshes, not tested in E4.
+ Float32 vs Float64 convergence study.

== Questions

- Is *Float32* acceptable for the applications you have in mind?
- *TPU access*: I currently pay for spot v5e myself (€0.49/h); could the group support a TPU
  Research Cloud application or GCP credits?
- Is the *upstream `P4estMesh` / KA path* the right scope (rather than `TreeMesh` / TrixiCUDA)?
- E4/E5 suggest the TPU wants *array operations, not raised kernels*: would a Reactant-specific
  array implementation of the volume and surface terms (package extension) be acceptable?
- Whom should I talk to about upstream changes — the owners of the GPU porting issue \#2822?
- Is the *AD angle* (Enzyme inside Reactant, as in Ariadne.jl) of interest for this project?

#show: appendix

= Appendix

== Repositories

- `junyixu/TrixiReactant.jl` (private): experiments E1–E4, `reactant_overrides.jl`, notes.
- `junyixu/TrixiDocker` (public): Arch-based GPU image with Trixi, Reactant, CUDA.jl,
  Enzyme and a separate TrixiCUDA environment, all precompiled; CUDA 12.9.
  - Reactant_jll's CUDA build links `libcuda.so.1`, so the GPU-less CI runner precompiles it
    against `CUDA_Driver_jll`'s forward-compatible `libcuda`.
