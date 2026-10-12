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
// glossary / footnote text for terms from outside numerical analysis
#let small(body) = block(above: 0.8em, text(size: 12.5pt, fill: luma(70), body))

#title-slide()

= Outline <touying:hidden>

#outline(title: none, indent: 1em, depth: 1)

= Question and background

== The question

*Can Trixi.jl run on TPUs through Reactant.jl, and how far does "run" go?*

#[
  #set text(size: 17pt)
  - A *TPU* is Google's accelerator chip, built around a matrix-multiply unit. It has no thread
    programming model like CUDA; the only way in is Google's compiler *XLA*, which takes programs
    made of whole-array operations.
  - *Reactant.jl* runs a Julia function once on placeholder ("traced") arrays, records each array
    operation into such a program (in *StableHLO*) and has XLA compile it for CPU, GPU or TPU.
  - Trixi's GPU code consists of *kernels* (KernelAbstractions.jl, "KA"): scalar code that each
    GPU thread runs for one index. For a TPU, Reactant must *raise* them into array operations.
]

#small[
  *XLA* (Accelerated Linear Algebra): the compiler behind TensorFlow and JAX; fuses array
  operations into machine code; the only compiler for TPUs. \
  *StableHLO* (HLO = High-Level Operations): XLA's input language — ≈ 100 whole-array operations
  (`add`, `dot_general` = matrix product, `slice`, `gather`, `while`, …) on fixed-size arrays, no
  threads, no pointers: one form of the computation that every XLA backend can optimize as a whole. \
  *MLIR* (Multi-Level Intermediate Representation, LLVM project): a toolkit for such compiler
  languages ("dialects") and the translations between them. StableHLO is one dialect; others
  describe loops and GPU kernels — that is how Reactant turns a kernel into StableHLO.
]

== What "raising" a kernel means

From Reactant's raising tutorial: a KA kernel and the output of `@code_hlo raise = true`.

#grid(
  columns: (1.1fr, 1fr),
  gutter: 1em,
  text(size: 15pt)[```julia
  @kernel function square_kernel!(y, @Const(x))
      i = @index(Global)        # thread i
      @inbounds y[i] = x[i] * x[i]
  end
  # launched with 64 threads, x has 64 entries
  ```],
  text(size: 15pt)[```
  func.func @main(%arg0: tensor<64xf64>)
                     -> tensor<64xf64> {
    %0 = stablehlo.multiply %arg0, %arg0
                     : tensor<64xf64>
    return %0 : tensor<64xf64>
  }
  ```],
)

#[
  #set text(size: 17pt)
  - 64 scalar multiplies (one per thread) become *one* vector multiply, which XLA fuses with the
    rest of the program and runs on any device. Unraised, the program just calls the CUDA kernel
    (NVIDIA GPUs only); on a TPU raising is switched on automatically.
  - Neighbour access `y[i] = x[nbr[i]]` becomes a `stablehlo.gather`: one memory lookup per
    index (E5, E6).
  - Data-dependent loops, small local arrays and shared memory do not raise yet (E2, E4).
]

So "Trixi on a TPU" means: every kernel Trixi launches must raise, and every data structure the
kernels touch must be an XLA array of plain numbers.

== Two existing GPU lines in the Trixi ecosystem

#table(
  columns: (auto, 1fr, 1fr),
  [], [*TrixiCUDA.jl*], [*Upstream Trixi.jl GPU path*],
  [Kernels], [hand-written CUDA.jl kernels re-implementing `rhs!`], [KA kernels inside Trixi (issue \#2822)],
  [Mesh], [`TreeMesh` 1D/2D/3D], [`P4estMesh` 2D/3D (`TreeMesh` "not planned")],
  [On the GPU], [semidiscretization only; time stepping = OrdinaryDiffEq on `CuArray`; some callbacks copy data to the CPU], [rhs, stepsize and analysis callbacks; `storage_type`, `real_type` options],
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
  [CFL-adaptive `dt`], [#ok], [`@trace while` with in-graph CFL reduction, ≈ 5% overhead (E7)],
  [callbacks with I/O], [CPU], [compile "advance k steps", do I/O on the CPU between calls],
  [AMR, p4est mesh generation], [#no], [XLA needs static shapes; p4est is a C library running on the CPU],
)

TrixiCUDA currently stops at the `rhs!` level; with Reactant the time loop itself can be compiled.

= Experiments

== Experiments at a glance

#show table: set text(size: 13pt)
#table(
  columns: (auto, 1.45fr, 1.2fr, auto),
  [], [*What I ran*], [*Question it answers*], [*Hardware*],
  [E1], [vector update and a sin/cos sum, Float32 and Float64, compiled by Reactant], [does Reactant use the device; what does Float64 cost?], [3070, TPU],
  [E2], [1D DG solver (Burgers) as hand-written CUDA kernels, TrixiCUDA style], [can GPU kernels be raised and run on a TPU?], [3070, CPU, TPU],
  [E3], [upstream Trixi's 2D P4est `rhs!` under Reactant], [does upstream Trixi compile unchanged?], [3070],
  [E4], [each of the 20 KA kernels of upstream Trixi alone, raised and not, 16 test cases], [which Trixi kernels raise, and what blocks the rest?], [A4500],
  [E5], [E2's solver in five variants (neighbours, data layout, array operations)], [why are raised kernels slow on the TPU?], [TPU],
  [E6], [reads through index arrays, blocks of 1–256 numbers per index], [cost of indirect access, i.e. of unstructured meshes], [TPU, A100],
  [E7], [2D Euler DG written as array operations, checked against Trixi's `rhs!`], [TPU speed vs Trixi on a GPU; Float32 convergence; adaptive `dt`], [TPU, A100, CPU],
  [E8], [upstream Trixi's own CUDA kernels, P4est 2D and 3D], [GPU baseline for E7 and E9], [A100],
  [E9], [3D version of E7], [does the picture hold in 3D?], [TPU, A100],
)

#small[
  GPUs rented on RunPod (RTX 3070, RTX A4500, A100); TPU: Google Cloud TPU v5e, one chip
  (`v5litepod-1`). Julia 1.12, Reactant 0.2.291, Trixi 0.17.14. "CPU, raised" = Reactant's CPU
  backend with `raise = true`, a stand-in for the TPU (the kernel must be raised there too).
]

== E1: device check and FP32 vs FP64

#table(
  columns: (auto, auto, auto, auto),
  [*Backend*], [*Type*], [*$y <- a x + y$, $n = 2^24$*], [*$sum_i (sin x_i cos x_i + x_i^2)^2$*],
  [RTX 3070], [Float32], [0.66 ms (305 GB/s)], [0.31 ms],
  [RTX 3070], [Float64], [1.22 ms (331 GB/s)], [5.17 ms],
  [CPU], [Float32], [5.13 ms], [1.16 ms],
  [CPU], [Float64], [9.39 ms], [3.48 ms],
)

- Reactant drives the 3070 at ≈70 % of peak memory bandwidth from plain Julia array code.
- Compute-bound work (the sin/cos sum) is *≈17× slower in Float64* on this consumer GPU.

== E2: TrixiCUDA-style kernels through Reactant

1D DGSEM, Burgers, LLF flux, periodic, $N = 3$, 5-stage low-storage RK, 100 steps. Kernels
written in TrixiCUDA style (`@cuda`, one thread per node, neighbours from index arrays).

#table(
  columns: (1.6fr, auto, auto, auto, auto),
  [*Mode (RTX 3070 unless noted)*], [*16k el. F32*], [*16k F64*], [*1M el. F32*], [*CPU backend*],
  [native CUDA.jl], [17.2 ms], [17.2 ms], [*256 ms*], [—],
  [Reactant, kernels not raised (run as CUDA code)], [9.9 ms], [13.7 ms], [252 ms], [7006 ms],
  [Reactant, kernels *raised* to StableHLO], [*3.1 ms*], [12.5 ms], [413 ms], [*79 ms*],
  [raised, volume kernel using shared memory], [#no], [#no], [#no], [#no],
)

All working modes reproduce the same-precision CPU reference *bit for bit* after 100 steps.

#small[
  *Shared memory*: a small, fast memory shared by the threads of one GPU thread block; the
  threads must wait at a barrier (`sync_threads`) before reading what the others wrote. Fast GPU
  kernels use it to reuse data, e.g. the two-point fluxes of flux differencing.
]

== E2: what we learned

- *Whole-loop compilation pays off for small problems*: one XLA program instead of a CPU loop
  launching ≈15 kernels per step → 5.5× faster than native CUDA.jl at 65k DOFs.
- *Raised code is not free*: at 4M DOFs it is 1.6× slower than the hand-written kernels.
- *Shared-memory kernels do not raise* ("failed to raise func") on GPU and CPU backends.
  TrixiCUDA's optimized kernels and Trixi's fastest 3D volume kernels use this pattern.
- Without raising, the CPU backend still runs the CUDA kernel (lowered to CPU code), but
  ≈90× slower than the raised version — and a TPU has no such fallback.

== E3: upstream Trixi through Reactant — blockers

Target: `p4est_2d_dgsem/elixir_advection_basic.jl`, `Trixi.rhs_hyperbolic!` under `@compile`. Not reached yet.

#table(
  columns: (1.2fr, 1fr),
  [*Blocker*], [*Workaround (method overrides only)*],
  [`wrap_array` uses `unsafe_wrap(pointer(u_ode))`], [return a `ReshapedArray` (Reactant writes back)],
  [containers build views with `unsafe_wrap_or_alloc` (pointers)], [reshaped device copy (no AMR)],
  [boundary/interface data stored as arrays of `Symbol` tuples], [keep on the CPU / encode as `Int8`],
  [kernels branch on `node_indices` with `===` on symbols], [integer index type + methods for the 5 index helpers],
)

In short: Trixi reaches its arrays through raw memory addresses (pointers) and stores some mesh
information as Julia `Symbol`s (names such as `:i_forward`); an XLA array has no addresses and
holds only numbers. On CUDA the last point works only because a `CuArray` can store pointers to
symbols. No Trixi source was changed — all workarounds are added methods.

== E4: inventory of Trixi's 20 KA kernels — method

- 16 cases with the elixirs and kernel variants of `test_cuda_2d/3d.jl`: advection, Euler
  (weak form, flux differencing with every sweep variant), MHD; 2D and 3D; Float32.
- A *recording backend* for KA stores every kernel launch (kernel and arguments) while Trixi
  computes the initial condition and one `rhs!` on the CPU; KA's CPU backend gives the
  reference results (native CUDA where that fails).
- Each kernel is then *replayed alone* through Reactant (`@compile raise = ...`); only its
  arguments are converted, which sidesteps the E3 blockers.
- Three modes: Reactant on the GPU *without* raising ("kept"), raised on the GPU, raised on the
  CPU (stand-in for the TPU). Output compared with the reference.
- No Trixi source changed; workarounds in the test harness on the next slides.

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

#small[
  Blockers: `scf.while` = a loop whose length the raiser cannot determine (it handles simple
  counted loops); stack `alloca` = a small local array inside the kernel; `sinpi` /
  `__nv_sinpif` = NVIDIA's `sinpi` routine, not translated; segfault = the compiler itself
  crashed; `@synchronize` = KA's barrier for shared memory. Sweep / turbo variants = Trixi's
  alternative flux-differencing kernels.
]

== E4: what this means

- *4 of 20 kernel definitions raise in every case, 2 in some, 14 never* — the 14 include
  every volume-term kernel, i.e. most of the arithmetic.
- The surface kernels that do raise contain *20–42 gathers and up to 36 scatters* each
  (reads `x[idx[i]]` and writes `y[idx[i]] = ...` through index arrays) — the pattern that is
  slow on a TPU (E5).
- Most blockers are compiler gaps, not Trixi design: loops the raiser does not recognise,
  small local arrays, a missing `sinpi`, a compiler crash, one wrong result.
- Shared memory: Reactant cannot raise barriers yet (stated in its own test suite). On a TPU
  shared memory is pointless anyway — XLA manages the on-chip memory itself — so Trixi's
  variants without shared memory are the ones to target.

== E4: bugs and gaps found (to report)

#table(
  columns: (auto, 1fr, auto),
  [*Where*], [*Issue*], [*Workaround in E4*],
  [Reactant], [kernel arguments that are *types* are rejected; every Trixi kernel receives `typeof(mesh)` (CUDA.jl accepts it)], [pass it as an empty value],
  [Reactant], [raised 2D interface kernel returns NaN in 80/4096 outputs (bitwise correct without raising)], [—],
  [Reactant], [compiler crash (segfault in one optimisation pass, `AffineCFGPass`) while raising 3 kernels], [skip kernel],
  [Reactant], [not raised yet: data-dependent loops, small local arrays, `sinpi`, barriers], [—],
  [Trixi], [3D HalfSweep kernels fail on KA's *CPU* backend: variables used across a barrier are not declared `@private`], [CUDA reference],
  [Trixi], [basis/mortar types require `AbstractMatrix{RealT}`; during tracing Reactant's arrays hold placeholders of type `TracedRNumber{RealT}`], [wrapper type],
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
- No matrix products in this program, so the TPU's reduced-precision matrix unit plays no role here.
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

- *Not the data layout*: XLA already stores the arrays with the element index running fastest,
  whichever order the Julia code uses.
- *The gathers*: raised kernels look up one number per index; run time follows the number of gathers.
- *As array operations the same solver is 57× faster* — and beats the 3070's native CUDA at
  4M DOFs. Results: kernels bitwise equal to the CPU; array form $4 times 10^(-6)$.

== E6: gathers on the TPU, block by block

64 MiB of Float32 split into blocks of $M$ numbers, read once through an index array
(`X[:, idx]`: column `idx[k]` of `X` for every $k$ — how an unstructured mesh finds its neighbours):

#table(
  columns: (1.7fr, auto, auto, auto, auto, auto),
  [*GB/s (copy: 365)*], [$M = 1$], [$4$], [$16$], [$64$], [$256$],
  [slices `X[:, 2:end]` (structured neighbour)], [164], [372], [375], [373], [360],
  [block gather `X[:, idx]`], [1.0], [5.2], [19], [41], [*234*],
  [block scatter], [1.2], [0.6], [9], [35], [113],
  [scalar gather (what raised kernels do)], [0.6], [0.3], [0.3], [0.3], [0.3],
  [A100: block gather, random indices], [130], [363], [577], [678], [671],
)

- *The TPU pays per index* (≈ 120 M indices/s, random or regular alike): only ≈ 1 KiB blocks
  come close to copy speed. Slices cost nothing — structured meshes are cheap.
- Unstructured meshes must gather *whole elements* (3D, $N = 3$: 320 numbers), not faces or nodes.

== E7: 2D Euler written as array operations

- Same discretization as Trixi (TreeMesh / Cartesian P4est): LGL $N = 3$, LLF, weak form *or*
  flux differencing (Kennedy–Gruber), convergence test with source terms, periodic.
- Volume term: weak form = one matrix product (`dot_general`) per direction; flux differencing =
  two-point fluxes of all node pairs by broadcasting, contracted with $D_"split"$. Neighbours:
  *slices* (structured) or *one block gather* of whole faces (unstructured).
- *Matches Trixi's `rhs!` to $10^(-12)$ in Float64* — all four variants, on the CPU, the TPU
  and an A100.
- Whole RK time loop in one XLA program, also with a CFL-adaptive `dt` computed inside it
  (`@trace while`, ≈ 5% overhead). Compile: ≈ 75 s the first time (Reactant warm-up), then
  3–13 s per configuration.

== E7: TPU v5e vs A100

`rhs!` in Float32, $2^10 times 2^10$ elements (16.8 M DOFs):

#table(
  columns: (1.9fr, auto, auto),
  [*Implementation*], [*weak form*], [*flux differencing*],
  [TPU v5e, array form, slices], [11.1 ms], [13.7 ms],
  [TPU v5e, array form, face gathers], [41.4 ms], [43.0 ms],
  [A100, same array code via Reactant], [*5.8 ms*], [*3.8 ms*],
  [A100, upstream Trixi CUDA kernels (P4est)], [16.4 ms], [13.7 ms],
)

- On the TPU the array form is *as fast as Trixi's own CUDA kernels on an A100*; on the A100
  it is 2–3.6× faster still (P4est handles curved meshes — not quite like for like).
- Cost per DOF at cloud prices: v5e ≈ A100 for the array code, both ≈ 3× cheaper than Trixi's
  current kernels. Face gathers: 3.7× slower on the TPU, free on the A100.

== E9: 3D — where Trixi's GPU kernels shine

`rhs!` in Float32, $64^3$ elements (16.8 M DOFs), same validation against Trixi ($10^(-11)$):

#table(
  columns: (1.9fr, auto, auto),
  [*Implementation*], [*weak form*], [*flux differencing*],
  [TPU v5e, array form], [37.0 ms], [49.2 ms],
  [A100, same array code via Reactant], [*12.9 ms*], [9.2 ms],
  [A100, upstream Trixi CUDA kernels (P4est)], [34.8 ms], [*5.2 ms*],
)

- Trixi's dedicated 3D shared-memory flux-differencing kernel is 1.8× faster than the array
  form on the A100 and *9.4× faster than the TPU* (≈ 3.6× cheaper per DOF at cloud prices).
- The TPU falls further behind in 3D: 3–5× slower than the A100 running the same code.
- Compile times stay modest: largest 3D program 17 s (`rhs!`) / 68 s (whole time loop).

== E7: precision on the TPU

$L^2$ error of $rho$, weak form, $t = 0.5$ (order 4 expected):

#table(
  columns: (1.6fr, auto, auto, auto, auto, auto),
  [*$2^L times 2^L$ elements, $L$ =*], [2], [3], [4], [6], [8],
  [Float32, `dot_general` DEFAULT (bf16)], [5.5e-3], [1.0e-2], [1.4e-2], [2.2e-2], [3.1e-2],
  [Float32, HIGH (bf16 × 3)], [8.4e-4], [5.6e-5], [3.3e-5], [2.1e-4], [3.0e-4],
  [Float32, HIGHEST], [8.4e-4], [5.1e-5], [*2.6e-6*], [3.7e-6], [1.1e-5],
  [Float64 (emulated, 49× slower)], [8.4e-4], [5.1e-5], [1.5e-6], [5.7e-9], [*2.8e-11*],
)

- *`dot_general` must run at HIGHEST* (only 8–17% slower): bf16 never converges.
- *Float32 stops converging at ≈ $3 times 10^(-6)$* (16×16 elements) and gets worse with
  refinement; flux differencing behaves the same. Float64 keeps order 4 down to $10^(-11)$.

#small[
  `dot_general` = XLA's matrix product. Its precision setting decides how the TPU's matrix unit
  multiplies Float32: DEFAULT rounds the inputs to bfloat16 (bf16, 8 significant bits ≈ 2–3
  digits), HIGH combines 3 bf16 products (≈ 16 bits), HIGHEST gives full Float32 accuracy.
]

= Precision

== Do we need Float64?

- *TPUs have no native Float64*: emulated on the v5e — 48× (weak form) and 103× (flux
  differencing) slower in E7, 170× in a compute-bound reduction (E1). On an A100 the same
  code pays only 1.6–2.3× for Float64.
- *Consumer GPUs*: FP64 at 1/64 of FP32 (17× slower measured on the 3070).
- *Upstream Trixi already supports Float32*: `semidiscretize(...; real_type = Float32)`;
  the CUDA/AMDGPU tests run Float32 with Float32 tolerances.
- *TPU pitfall*: XLA computes Float32 matrix products in bfloat16 (≈ 2–3 digits) by default,
  which destroys convergence (E7); always
  `Reactant.with_config(; dot_general_precision = PrecisionConfig.HIGHEST)`.

== Float32: what we found

- *Accuracy floor*: with Float32 the $L^2$ error of the 2D Euler convergence test bottoms out
  at ≈ $3 times 10^(-6)$ and grows again under refinement (more steps, more cancellation in
  $J^(-1) (dots)$) — fine for engineering accuracy, not for convergence studies.
- Single `rhs!` evaluations look worse ($10^(-5)$ – $10^(-4)$ relative vs Float64) than the
  solution, as already seen in E2.
- Still to check: long-time runs, entropy-conservative fluxes (log means), steady states.
- *Question for you*: are the target applications fine with errors around $10^(-6)$?

= Summary

== Assessment so far

- *Trixi's KA kernels*: all 20 run through Reactant on a GPU unchanged, but only 4 raise in
  every case — today there is no TPU path by raising Trixi's kernels (E4).
- *TPUs want array operations*: an indirect access costs ≈ 8 ns per index (E5, E6). Written as
  array code, 2D DG on one v5e matches Trixi's CUDA kernels on an A100 (E7); 3D does not (E9).
- *Precision*: Float32 caps the error at ≈ $3 times 10^(-6)$ on this test (CPU and TPU alike);
  Float64 is 48–103× slower on the TPU; `dot_general` must run at HIGHEST.
- *Cost*: at cloud prices a v5e ≈ an A100 for the 2D array code — but the A100 also runs the
  existing Trixi code and fast Float64.
- *My reading*: TPUs are worth it only for Float32-tolerant runs on structured or
  element-blocked meshes with an array implementation; for Trixi in general, GPUs remain the
  better target.

= Next steps

== Plan

+ *Report the Reactant issues* with minimal examples: type-valued kernel arguments, the raised
  kernel with NaN results, the compiler crash, and what does not raise yet (data-dependent
  loops, `sinpi`).
+ *Array formulation beyond Cartesian meshes*: curved P4est meshes (metric terms) and
  unstructured connectivity via *element-sized* gathers (E6), then 3D. Packaged as a Trixi
  package extension (code that is loaded only together with Reactant) that swaps in array
  versions of the volume and surface terms.
+ Runs on 4–8 TPU chips (Reactant splits the arrays across chips) once single-chip numbers are
  settled.
+ Report the Trixi issues (HalfSweep kernels on KA's CPU backend, basis type constraints,
  `Symbol`-valued `node_indices`) on Trixi's GPU-porting issue \#2822; mortars /
  non-conforming meshes are still untested.

== Questions

#[
  #set text(size: 17pt)
  - Is *Float32* acceptable for the applications you have in mind?
  - Is the *upstream `P4estMesh` / KA path* the right scope (rather than `TreeMesh` / TrixiCUDA)?
  - *Shared memory*: Trixi's fastest 3D flux-differencing kernels use it, and Reactant cannot
    raise them (no support for barriers yet); I have not worked on this. Options: (a) help add
    barrier support to Reactant — compiler work; (b) target Trixi's variants without shared
    memory — today blocked by a compiler crash and a loop that does not raise; (c) leave
    raising and write array implementations (next question). On a TPU shared memory brings no
    speed-up anyway, since XLA manages the on-chip memory. Which would you prefer?
  - Most of Trixi's kernels do not raise, and the ones that do are slow on the TPU because of
    their index lookups, while the same solver written as array operations is fast. Would a
    Reactant-specific *array implementation* of the volume and surface terms (package
    extension) be acceptable?
  - Is *automatic differentiation* of interest for this project — Enzyme inside Reactant,
    differentiating the compiled solver (as in Ariadne.jl)?
]
