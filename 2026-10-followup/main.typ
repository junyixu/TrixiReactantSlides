#import "@preview/touying:0.6.1": *
#import themes.metropolis: *

#show: metropolis-theme.with(
  aspect-ratio: "16-9",
  footer: self => self.info.title,
  config-info(
    title: [Trixi.jl on TPUs via Reactant.jl],
    subtitle: [Follow-up: P4est array prototype — kernel raising, coverage, speed],
    author: [Junyi Xu],
    date: datetime(year: 2026, month: 10, day: 18),
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

= Since the last meeting

== Decisions of 13 October and what I did

#table(
  columns: (1.25fr, 1.9fr, auto),
  [*Decision*], [*What I did*], [],
  [array implementation, first as a separate prototype],
  [*E12*: the DG `rhs!` as array operations for `P4estMesh`, with Trixi's own pointwise physics
    functions; extended in E14–E25],
  [#ok],
  [`P4estMesh` in 2D and 3D, 3D more important],
  [3D (*E12*) and 2D (*E16*, *E20*): curved, unstructured meshes; mortars (*E18*, *E25*), AMR (*E21*)],
  [#ok],
  [Float32; test cases from Trixi's GPU tests],
  [all 3D and 2D GPU test cases run to the end on the TPU, within their Float32 tolerances (*E14*–*E16*)],
  [#ok],
  [one of HalfSweep / FullSweep / FullSweepGlobal through Reactant],
  [*E11*: FullSweepGlobal raises and is correct on a TPU],
  [#ok],
  [`Symbol` → enum (Trixi PR \#3332)],
  [*E13*: the PR tested with the kernel inventory (E4) and the whole `rhs!` (E3)],
  [#ok],
  [Reactant first: it will expose low-level problems],
  [9 issue reports drafted with minimal examples, re-tested on Reactant 0.2.297, not posted yet],
  [#warn],
  [question: the same speed by changing KA kernels, without Reactant?],
  [*E17*: on an A100 the E11 changes make the kernel 7–9× slower; Trixi's HalfSweep is already fast],
  [#ok],
)

#small[
  *Mortars*: Trixi's coupling across faces between elements of different size (non-conforming meshes,
  AMR); a *conforming* mesh has none. \
  Hardware: one TPU v5e chip (four in E24); NVIDIA L4 (E11) and A100 (E17). Float32 unless noted.
]

== Experiments since 16 October

#[
  #show table: set text(size: 13.5pt)
  #table(
    columns: (auto, 1.25fr, 2fr),
    [*Exp.*], [*Topic*], [*Result*],
    [E14], [whole simulations; time loop], [GPU test cases within Float32 tolerances; one program per RK stage],
    [E15], [GLM-MHD], [Trixi's MHD functions unchanged; to round-off],
    [E16, E20], [2D `P4estMesh`], [to round-off; brick path 12× faster than the gather],
    [E17], [A100], [array code: weak form 2.8× faster than Trixi's kernels, flux diff. 2.3× slower],
    [E18], [mortars], [to round-off with 298 mortars on a curved unstructured mesh],
    [E19], [JAX in the image], [a Pallas kernel runs inside a Reactant program on the TPU, exactly],
    [E21], [AMR], [adapted meshes reuse one compiled program],
    [E22], [shock capturing], [to round-off; 10–13% more than flux differencing],
    [E23], [Pallas volume kernel], [12.6× faster than XLA; flux-differencing time step 1.73× faster],
    [E24], [4 TPU chips], [linear scaling, bitwise identical results],
    [E25], [2D mortars, shock capturing, AMR], [to round-off; whole runs and AMR runs also match Trixi's test values],
    [E26], [whole RK stage as one Pallas kernel], [18.2 ms per RK step, faster than Trixi on an A100 (≈ 33 ms)],
  )
]

#small[*To round-off*: `rhs!` and whole Float64 simulations agree with Trixi to $≈ 10^(-12)$ or better.]

= E11: Trixi's flux-differencing kernel on the TPU

== E11: what blocked raising

*FullSweepGlobal* is Trixi's 3D flux-differencing kernel without shared memory: one GPU thread
per node. HalfSweep and FullSweep synchronize threads (barriers), which Reactant cannot raise yet.

#[
  #set text(size: 17pt)
  + *KA's bounds check*: every thread first tests whether its index lies in the 4D index range.
    Julia compiles that test into a loop over a small stack array, which Reactant cannot raise —
    so *no* KA kernel with a 4D index range raises. Fix: an extra unrolled `in` method (no change
    to Trixi or KA), or `@kernel unsafe_indices = true`, which drops the check.
  + *A Reactant compiler crash* (in an integer-remainder optimization): avoided with a fixed
    workgroup size and unrolled node loops.
  + *Node loops become gathers*: in the original `for ii in 1:n` loops, `u[v, ii, j, k, element]` is
    an indexed read — 244 gathers. Compiling that for the TPU needed more than 33 GB of host
    memory. With the three node loops unrolled ($n$ known at compile time): no gathers.
]

#small[
  *KA*: KernelAbstractions.jl, in which Trixi writes its GPU kernels. *Raise*: Reactant turns a
  kernel into whole-array operations, the only form XLA can compile for a TPU. *Gather*: an
  indexed read; on the TPU ≈ 8 ns per index (E6).
]

== E11: results

#grid(
  columns: (1fr, 1.15fr),
  gutter: 1em,
  [
    NVIDIA L4, 64 elements, one call:
    #table(
      columns: (1.6fr, auto),
      [*FullSweepGlobal*], [*time*],
      [Trixi's kernel, not raised], [0.191 ms],
      [Trixi's kernel, raised (244 gathers)], [0.400 ms],
      [node loops unrolled, raised], [*0.153 ms*],
    )
  ],
  [
    TPU v5e, node loops unrolled, raised:
    #table(
      columns: (auto, auto, auto, auto),
      [*Elements*], [*Nodes*], [*Error*], [*Time*],
      [64], [4 k], [$1.3 times 10^(-7)$], [0.19 ms],
      [4,096], [262 k], [$3.5 times 10^(-7)$], [0.39 ms],
      [32,768], [2.1 M], [$2.6 times 10^(-7)$], [4.3 ms],
      [262,144], [16.8 M], [$3.5 times 10^(-7)$], [47.8 ms],
    )
  ],
)

#[
  #set text(size: 17pt)
  - Correct on the TPU (error relative to Trixi's own kernel). Changes to Trixi's kernel: unroll
    the three node loops, fix the workgroup size, drop the bounds check.
  - *But slow there*: at 16.8 M nodes this one kernel takes as long as the whole flux-differencing
    `rhs!` of the array code (E9: 49.2 ms, 22.7 ms on a 128-friendly grid); Trixi's whole `rhs!`
    on an A100 takes 5.2 ms.
  - Profile: XLA splits the raised kernel into ≈ 240 kernels that read and write ≈ 100
    state-sized arrays. 4,096 elements stay in on-chip memory; 262,144 go through device memory.
]

#small[
  *State-sized array*: as large as the solution (5 variables at every node). A `rhs!` needs about
  2.5 such reads/writes at minimum: read the state and the geometry, write $partial_t u$.
]

= E12: array prototype on P4est meshes

== E12: how the prototype works

#[
  #set text(size: 16.5pt)
  Input: any conforming 3D `P4estMesh` set up by Trixi (curved, unstructured, Dirichlet
  boundaries); mesh data taken once from Trixi's data structures.
  - *Physics from Trixi*: `flux`, the two-point flux (`flux_kennedy_gruber`), the surface flux
    (`FluxLaxFriedrichs`), boundary values and source terms are Trixi's own scalar functions,
    applied elementwise to whole arrays. Other equations need no new flux code (tested: Euler 3D).
  - *DG operators as array operations* on one $(n, n, n, "elements")$ array per variable: weak
    form or flux differencing along each node dimension, metric terms taken from Trixi.
  - *Neighbour values*, three variants:
    - *gather*: one indexed read of whole faces (all variables, $n^2$ nodes), then one of 8 face
      orientations — any conforming mesh;
    - *slices*: a brick-shaped mesh reordered into one $N_x times N_y times N_z$ grid of
      elements — no indexed reads;
    - *tree blocks*: after uniform refinement each tree is an $m times m times m$ block of
      elements — slices inside a tree, one gather of whole tree faces between trees.
]

#small[*Brick-shaped*: the trees form a Cartesian arrangement, all face orientations are the identity.]

== E12: validation against Trixi's `rhs!`

Relative max. difference of $partial_t u$ from Trixi's `rhs!`, at the exact solution, $t = 0.3$:

#table(
  columns: (1fr, 2fr, auto, auto, auto),
  [*Case*], [*Mesh*], [*Volume term*], [*Float64*], [*Float32*],
  table.cell(rowspan: 2)[Cartesian (3D GPU test)#super[a]],
  table.cell(rowspan: 2)[$2^3$ trees refined once: 64 elements, $n = 4$, Dirichlet on all sides],
  [weak form], [$1.3 times 10^(-13)$], [$4.1 times 10^(-5)$],
  [flux diff.], [$1.3 times 10^(-13)$], [$5.6 times 10^(-5)$],
  table.cell(rowspan: 2)[HOHQMesh#super[b]],
  table.cell(rowspan: 2)[curved, unstructured: 204 elements, $n = 5$, 96 faces with rotated orientation],
  [weak form], [$1.4 times 10^(-13)$], [$7.3 times 10^(-5)$],
  [flux diff.], [$2.1 times 10^(-13)$], [$1.0 times 10^(-4)$],
)

- Float64 agrees to round-off: the same discretization, not an approximation of it. The Float32
  differences are Float32 round-off (≈ $10^(-5)$ in E7 as well).
- The same on the TPU (Float64 is emulated there): ≤ $7 times 10^(-12)$. All three neighbour
  variants and every combination of the optimizations below were checked this way.

#small[
  Trixi's 3D P4est examples, compressible Euler with source terms: #super[a]
  `elixir_euler_source_terms_nonperiodic.jl`, #super[b] `elixir_euler_source_terms_nonperiodic_hohqmesh.jl`.
]

== E12: six optimization ideas, tried one by one

TPU v5e, Float32, 262,144 elements (16.8 M nodes), speed-up of `rhs!` over the previous step:

#table(
  columns: (1.5fr, 1.4fr),
  [*Idea*], [*Effect*],
  [1. derivatives as $n$ slice terms instead of transpose → matrix product → transpose],
  [`rhs!` unchanged; compiled time loop 2.7× faster],
  [2. flux differencing without arrays of all node pairs],
  [unchanged (7–18% with one call for all 5 components)],
  [3. face fluxes straight from slices (brick meshes)], [*1.8–3.6×*: no relayout of face arrays],
  [4. fastest array dimension a multiple of 128], [*1.8–5×*; in the time loop it must be the *last* dimension],
  [5. tree blocks (unstructured meshes)], [correct, but slower as written: tree faces are copied at 38 GB/s],
  [6. fused Runge–Kutta stage update], [no effect],
)

#small[
  The TPU stores arrays in tiles 128 wide along the fastest-varying dimension and pads a shorter
  one. For a single `rhs!` XLA chooses that dimension itself; inside a compiled time loop it takes
  the *last* one. With $16 times 16 times 128$ elements instead of $128 times 16 times 16$, an RK
  step drops from 119 ms to 24 ms (32,768 elements, weak form). Each idea was profiled on the TPU
  before the next one.
]

== E12: where it stands

#table(
  columns: (2.1fr, auto, auto),
  [*`rhs!`, 262,144 elements (16.8 M nodes), Float32*], [*weak form*], [*flux differencing*],
  [TPU, first version], [251 ms#super[a]], [—],
  [TPU, best `rhs!` ($128 times 64 times 32$ elements)], [38.9 ms], [55.7 ms],
  [TPU, best time step ($32 times 64 times 128$ elements)], [55.0 ms (RK step 230 ms)], [75.0 ms],
  [TPU, Cartesian-only array code (E9), $128 times 64 times 32$], [18.9 ms], [22.7 ms],
  [A100, Trixi's CUDA kernels (E8, $64^3$ elements)], [34.8 ms], [*5.2 ms*],
)

#[
  #set text(size: 17pt)
  - 6.5× faster than the first version; still 2× behind the Cartesian-only code of E9, and 11×
    behind Trixi's flux-differencing kernel on an A100.
  - The best version, like the raised kernel of E11, still reads and writes 78–110 state-sized
    arrays per `rhs!` (≈ 2.5 needed): hundreds of small kernels, 150–190 transposes. Rewriting
    the expressions barely changed this; the gains came from the layout.
  - Flux differencing inside the compiled time loop: 12–32 min compile time and 4× slower than
    five separate `rhs!` calls. Cause found in E14.
]

#small[
  #super[a] The compiled time loop does not fit into the 16 GB device memory of a v5e. \
  An RK step is 5 `rhs!` calls (low-storage, 4th order, 5 stages), compiled as one program.
]

== E12: curved unstructured meshes

HOHQMesh case refined uniformly, $n = 5$, gather vs tree blocks:

#table(
  columns: (auto, auto, auto, auto),
  [*Mesh*], [*Neighbours*], [*weak form `rhs!` (RK step)*], [*flux diff. `rhs!`*],
  table.cell(rowspan: 2)[13,056 elements (1.6 M nodes)],
  [gather], [*17.3 ms* (78.5 ms)], [*20.7 ms*],
  [tree blocks, $4^3$], [23.2 ms (142 ms)], [36.4 ms],
  table.cell(rowspan: 2)[104,448 elements (13 M nodes)],
  [gather], [164 ms#super[a]], [—],
  [tree blocks, $8^3$], [*118 ms* (868 ms)], [191 ms],
)

- ≈ 9–11 ns per node and `rhs!` on curved unstructured meshes, vs 2.3 ns on brick meshes (first
  version: 15 ns).
- Tree blocks pay off only for large trees ($8^3$: 1.4× faster than the gather). As written, every
  tree face is rearranged into an $(n m, n m, "trees")$ array, a slow copy; a representation
  without that copy is the next thing to try.

#small[#super[a] Compiling the time loop ran out of host memory.]

= E13: enums instead of symbols

== E13: Trixi PR \#3332 with Reactant

The PR stores `node_indices` (how the nodes of an interface map to each of its elements) as tuples
of a one-byte enum `NodeIndex.T` instead of `Symbol`s. Tested with a Trixi build from the PR:

#[
  #set text(size: 17pt)
  - *Kernel inventory* (E4: each Trixi KA kernel replayed and raised alone): Trixi's index helpers
    now work inside raised kernels as they are — before, I replaced 5 of them with integer
    versions. All 11 test cases give the same results as before.
  - One adapter remains: an *array of tuples* cannot be an XLA buffer, so it becomes an integer
    array with an extra leading dimension, turned back into tuples on indexing.
  - *Reactant bug*: that array as `UInt8` (the enum's type) breaks every kernel that receives it;
    the same bytes as `Int8` work.
  - *Whole `rhs!`* (E3, 2D P4est): Trixi's conversion to device arrays (`trixi_adapt`) now succeeds
    (it failed on the symbols). Next blockers: Trixi's containers require `DenseArray{RealT}`,
    which traced Reactant arrays are not, and a second Reactant bug. Not pursued — the array
    route does not need this.
]

#small[
  For the PR this suggests `Int8` as the enum's base type, or the tuple-to-integer conversion as an
  adapt rule in Trixi's Reactant extension.
]

= E14, E15: whole simulations

== E14: the GPU test cases run to the end

Carpenter–Kennedy RK as in the elixirs, the time step from the CFL number in every step (Trixi's
`max_dt`), errors from Trixi's own `calc_error_norms`. TPU v5e, Float32; max. relative difference
of l2 / linf:

#table(
  columns: (1.6fr, auto, 1fr, 1fr),
  [*Case*], [*Steps (Trixi)*], [*vs Trixi, CPU, Float64*], [*vs GPU test reference*],
  [3D advection, 512 elements], [19 (19)], [$8.3 times 10^(-6)$ / $5.2 times 10^(-4)$],
  [$8.3 times 10^(-6)$ / $5.2 times 10^(-4)$],
  [3D Euler, weak form, 64 elements], [317 (317)], [$1.8 times 10^(-4)$ / $2.4 times 10^(-4)$], [not in the tests],
  [3D Euler, flux differencing], [317 (317)], [$1.3 times 10^(-4)$ / $1.3 times 10^(-3)$],
  [$2.0 times 10^(-4)$ / $1.5 times 10^(-3)$],
  [3D MHD, 512 elements (E15)], [84 (84)], [$1.1 times 10^(-4)$ / $3.3 times 10^(-4)$],
  [$1.1 times 10^(-4)$ / $3.3 times 10^(-4)$],
  [2D advection, Euler, MHD (E16)], [18, 229, 36 (same)], table.cell(colspan: 2)[within tolerance#super[a]],
)

- All within TrixiTest's Float32 tolerances (atol $≈ 6 times 10^(-5)$, rtol $≈ 3.5 times 10^(-4)$),
  with the same number of steps. Our differences from Trixi are as large as Trixi's own differences
  between Float32 and Float64.

#small[
  #super[a] 2D Euler is at the Float32 floor: Trixi's own Float32 and Float64 l2 errors differ by
  1.5–1.8×, so only the Float64 `rhs!` comparison (≤ $6 times 10^(-13)$) says much there.
]

== E14: how to compile the time loop

262,144 elements, 10 RK steps with a fixed $Delta t$; time per RK step (in brackets: relative to
five separate `rhs!` calls):

#table(
  columns: (2fr, 1fr, 1fr),
  [*Program*], [*weak form*], [*flux differencing*],
  [5 × `rhs!`, separately], [195 ms], [279 ms],
  [whole simulation in one program (`@trace`, E12)], [300 ms (1.54)], [compiles for > 1 h],
  [one program per RK step], [199 ms (1.02)], [out of device memory#super[a]],
  [one program per RK stage, loop in Julia], [*197 ms (1.01)*, compile 13 s], [*283 ms (1.01)*, compile 27 s],
)

- The slow flux-differencing time loop of E12 comes from *five flux-differencing `rhs!` in one XLA
  program* (in another layout 3.3× slower and 30 min compile time), not from the loop. The same on
  an A100 (6×, E17): an XLA fusion problem, not a TPU one.
- One program per stage compiles in seconds, costs exactly 5 `rhs!` per step; calling it from Julia
  costs nothing measurable.

#small[#super[a] With $32 times 64 times 128$ elements it fits: 1226 ms (3.26×) after 1778 s of compiling.]

== E15: GLM-MHD with Trixi's own functions

9 variables, Powell non-conservative terms ("combined" fluxes), HLL surface flux, a cleaning speed
$c_h$ that changes every step (GPU test `elixir_mhd_alfven_wave_combined_fluxes_nonperiodic.jl`).
Trixi's MHD functions are used unchanged; three things were needed:

#[
  #set text(size: 17pt)
  + *Combined fluxes* return a left and a right part: each element face takes the left part, with
    its own state first and its own outward normal.
  + *Functions that branch on values* (`ln_mean`, `FluxHLL`): branch-free versions with `ifelse`
    (Trixi already has such versions for SIMD types).
  + *$c_h$ cannot be a field of Trixi's equations* (fields must be `Real`; a traced number is not):
    the fluxes are affine in $c_h$, so $f = f(c_h = 0) + c_h (f(c_h = 1) - f(c_h = 0))$.
]

Result: `rhs!` vs Trixi in Float64 $7.6 times 10^(-14)$ (3D), $4.5 times 10^(-14)$ (2D); whole runs
within the GPU tests' tolerances, with the same number of steps.

#small[These are Trixi patterns that any tracing approach runs into (issue draft 8).]

= Coverage: 2D, mortars, AMR, shock capturing

== E16, E25: 2D, validation against Trixi's `rhs!`

The same code as in 3D, with 4 faces per element and 2 face orientations (forward, backward)
instead of 8. Relative max. difference of $partial_t u$ from Trixi's `rhs!` at the exact solution:

#table(
  columns: (auto, 1.9fr, auto, auto, auto),
  [*Case*], [*Mesh*], [*Volume term*], [*Float64*], [*Float32*],
  [advection#super[a]], [$8 times 8$ trees, 256 elements, periodic], [weak form], [$8.1 times 10^(-14)$],
  [$2.3 times 10^(-5)$],
  table.cell(rowspan: 2)[Euler#super[a]], table.cell(rowspan: 2)[$16 times 16$ trees, 256 elements, periodic],
  [weak form], [$2.7 times 10^(-13)$], [$1.1 times 10^(-4)$],
  [flux diff.], [$5.9 times 10^(-13)$], [$2.1 times 10^(-4)$],
  [MHD#super[a]], [64 elements, Dirichlet], [flux diff.], [$4.5 times 10^(-14)$], [$1.8 times 10^(-5)$],
  [advection#super[b]],
  table.cell(rowspan: 3)[curved, unstructured: 384 elements, 32 of 736 interfaces with reversed node order, Dirichlet],
  [weak form], [$1.7 times 10^(-13)$], [$6.7 times 10^(-5)$],
  table.cell(rowspan: 2)[Euler#super[c]], [weak form], [$3.5 times 10^(-13)$], [$1.8 times 10^(-4)$],
  [flux diff.], [$8.7 times 10^(-13)$], [$3.6 times 10^(-4)$],
)

#[
  #set text(size: 17pt)
  - Whole simulations on the TPU (Float32): the curved unstructured cases agree with Trixi's Float64
    runs to $≤ 3 times 10^(-5)$ (l2), the 2D GPU tests pass their Float32 tolerances.
]

#small[
  #super[a] Trixi's 2D GPU tests (`test_cuda_2d.jl`). #super[b] `elixir_advection_unstructured_flag.jl`;
  #super[c] `elixir_euler_source_terms_nonconforming_unstructured_flag.jl` without its corner
  refinement. Both: an unstructured quadrilateral mesh read from a file, curved by a mapping.
]

== E20, E25: 2D, speed

TPU v5e, Float32, $1024 times 1024$ elements (16.8 M nodes, as in E7), Euler, `rhs!`:

#table(
  columns: (2.2fr, auto, auto),
  [*Variant*], [*weak form*], [*flux differencing*],
  [TPU, gather (any conforming mesh, E16)], [227 ms], [229 ms],
  [TPU, brick path: faces from slices (E20)], [*18.3 ms* (RK step 93 ms)], [*27.2 ms* (142 ms)],
  [TPU, Cartesian-only array code (E7)], [11.1 ms], [13.7 ms],
  [A100, Trixi's CUDA kernels (E8)], [16.4 ms], [13.7 ms],
)

#[
  #set text(size: 17pt)
  - The brick path of the 3D prototype (E12, idea 3) now works in 2D and with periodic directions:
    12× faster than the gather; the weak form is on par with Trixi on an A100, flux differencing
    2× behind.
  - Unstructured meshes need the gather: on the curved mesh above refined to 1.6 M elements, 389 ms
    (weak form) and 392 ms (flux differencing), ≈ 15 ns per node and `rhs!` (3D: 9–11 ns).
]

#small[
  2D has more face nodes per element than 3D (12 of 16 at $n = 4$), so faces weigh more; a gather on
  the TPU costs per index (E6).
]

== E18: mortars (non-conforming meshes)

A small face takes the L2 projection of the large face as its neighbour value; the large face gets
$-4 sum$ (reverse projection) of the mortar fluxes ($-2 sum$ in 2D). *Upstream Trixi's GPU path has
no mortars*; the reference is Trixi's default CPU path. `rhs!` vs Trixi:

#table(
  columns: (auto, 2fr, auto, auto, auto),
  [*Case*], [*Mesh*], [*Volume term*], [*Float64*], [*Float32*],
  [3D advection], [71 elements, one corner refined, 6 mortars], [weak form],
  [$3.6 times 10^(-13)$], [$1.1 times 10^(-4)$],
  table.cell(rowspan: 2)[3D Euler], table.cell(rowspan: 2)[curved, unstructured, 1020 elements, $n = 5$, 298 mortars],
  [weak form], [$1.7 times 10^(-12)$], [$7.0 times 10^(-4)$],
  [flux diff.], [$2.6 times 10^(-12)$], [$1.6 times 10^(-3)$],
  [2D advection (E25)], [curved, 168 elements, 72 mortars], [weak form], [$2.0 times 10^(-13)$],
  [$1.2 times 10^(-4)$],
  table.cell(rowspan: 2)[2D Euler (E25)], table.cell(rowspan: 2)[curved, unstructured, 168 elements, 80 mortars],
  [weak form], [$1.6 times 10^(-13)$], [$6.4 times 10^(-5)$],
  [flux diff.], [$2.9 times 10^(-13)$], [$9.4 times 10^(-5)$],
)

#[
  #set text(size: 17pt)
  - Whole runs, Float64 vs Trixi: $≤ 10^(-12)$ (3D curved Euler: 162 steps; 2D: 221 steps). Float32 on
    the TPU: $≤ 1.8 times 10^(-4)$ in 2D; $1.7 times 10^(-4)$ / $2.7 times 10^(-3)$ for the 3D curved
    Euler (round-off amplified by small Jacobians). The 2D elixirs also match Trixi's test values.
]

== E21: AMR without recompiling

The gather path takes the mesh as array data; padded to a capacity (2× its size), an adaptation only
changes array contents, not shapes. Adapting itself is Trixi's `AMRCallback`, on the CPU.
SC: shock capturing; l2 / linf, max. relative difference:

#[
  #set text(size: 13.5pt)
  #table(
    columns: (2.4fr, auto, auto, auto, auto, auto),
    [*Elixir*], [*Steps*], [*Adapt.*], [*Compil.*], [*Float64 vs Trixi*], [*TPU Float32 vs Trixi*],
    [3D advection, $t = 3$ (≈ 6,000 elements)], [90], [17], [1],
    [$7.2 times 10^(-15)$ / $2.2 times 10^(-13)$], [$3.6 times 10^(-5)$ / $5.2 times 10^(-4)$],
    [3D blast wave + SC, adapting every step (≈ 1,000 el.)], [42], [40], [1],
    [$5.2 times 10^(-16)$ / $1.6 times 10^(-14)$], [$1.2 times 10^(-5)$ / $2.5 times 10^(-4)$],
    [2D blast wave + SC (64–196 elements, E25)], [79], [15], [2],
    [$1.5 times 10^(-15)$ / $1.0 times 10^(-14)$], [$4.9 times 10^(-5)$ / $1.1 times 10^(-4)$],
    [2D advection, curved unstructured, $t = 10$ (≈ 200 el., E25)], [184], [36], [1],
    [$1.0 times 10^(-15)$ / $7.9 times 10^(-16)$], [$1.6 times 10^(-5)$ / $7.3 times 10^(-7)$],
  )
]

#[
  #set text(size: 16pt)
  - Same mesh as Trixi after every adaptation (Float32: the 3D blast wave ends with 7 elements fewer);
    the 2D blast wave's start mesh outgrew its capacity once (2 compilations). Each adaptation is a host round trip of 0.03–1.3 s, against 5–61 ms per RK step on the TPU: the
    host part dominates (next: indicator and projection on the device).
]

== E22: shock capturing

Trixi's `VolumeIntegralShockCapturingHG`: per element a blending factor $alpha$ from the modal energy
of an indicator variable, smoothed over neighbours; volume term $(1 - alpha)$ flux differencing +
$alpha$ first-order finite volumes on the subcells. All array operations, no branch per element.

#grid(
  columns: (1.25fr, 1fr),
  gutter: 1em,
  [
    Sedov blast wave (Trixi's test, $t = 0.3$):
    #table(
      columns: (1.3fr, auto, auto),
      [], [*Float64*], [*Float32*],
      [3D `rhs!` (64 el., $n = 6$)], [$2.4 times 10^(-16)$], [$1.0 times 10^(-7)$],
      [3D run, 28 steps (TPU for Float32)], [$1.6 times 10^(-16)$], [$3.2 times 10^(-7)$],
      [2D `rhs!` (256 el., $n = 5$, E25)], [$1.9 times 10^(-16)$], [$3.6 times 10^(-8)$],
      [2D run, 103 steps (E25)], [$2.0 times 10^(-15)$], [$6.8 times 10^(-5)$],
    )
  ],
  [
    TPU, 3D `rhs!`, cost:
    #table(
      columns: (auto, auto, auto),
      [*Elements*], [*shock capt.*], [*flux diff.*],
      [4,096], [7.31 ms], [6.65 ms],
      [32,768], [85.5 ms], [75.7 ms],
    )
  ],
)

- Float64 agrees with Trixi almost bit for bit: the array code does the same operations in the
  same order. Shock capturing costs 10–13% more than plain flux differencing (both FD and FV are
  computed in every element).
- With AMR: the blast wave on the previous slide.

= Speed: GPU comparison, hand-written kernels, several chips

== E17: the comparison on an A100

262,144 elements (16.8 M nodes), Float32:

#table(
  columns: (2.1fr, auto, auto, auto),
  [*`rhs!`*], [*weak form*], [*flux diff.*], [*RK step*],
  [A100, Trixi's CUDA kernels], [36.0 ms], [*6.5 ms*], [—],
  [A100, array prototype (Reactant → CUDA)], [*12.9 ms*], [14.7 ms], [67 / 99 ms],
  [TPU v5e, array prototype (E12, E14)], [38.9 ms], [55.7 ms], [197 / 283 ms],
  [TPU v5e, with the Pallas volume kernel (E23)], [—], [32.3 ms], [— / 164 ms],
  [TPU v5e, whole RK stage in one Pallas kernel (E26)], [—], [3.1 ms], [— / *18.2 ms*],
)

#[
  #set text(size: 17pt)
  - On a GPU the array code beats Trixi's kernels for the weak form (2.8×) and loses for flux
    differencing (2.3×: Trixi's HalfSweep computes each two-point flux once, in shared memory).
  - Your question, changing KA kernels instead of using Reactant: the flux-differencing volume term
    alone takes 2.11 ms (HalfSweep), 3.95 ms (FullSweepGlobal), but 26–29 ms with E11's unrolled
    node loops — they only help raising. On a GPU, the kernel worth improving is Trixi's weak-form
    volume kernel (one thread per element, 78–85% of its `rhs!`, E10).
]

#small[RK step: weak form / flux differencing, one compiled program per stage (E14); E26: one program for all steps.]

== E23: a hand-written TPU kernel for the volume term

E19: a Pallas kernel lowered by JAX runs inside a Reactant program on the TPU, exactly. E23: the 3D
flux-differencing volume term (Kennedy–Gruber, curved metric terms) as one Pallas kernel — all
intermediate values stay in on-chip memory.

#table(
  columns: (2fr, 1fr, 1.3fr),
  [*262,144 elements, Float32*], [*XLA*], [*Pallas*],
  [volume term alone (pure JAX)], [27.6 ms], [*2.19 ms* (12.6×; ≈ 570 GB/s)],
  [flux-differencing `rhs!` in the prototype], [55.8 ms], [*32.3 ms*],
  [RK step], [282.6 ms], [*163.5 ms* (1.73×)],
  [device-memory traffic per `rhs!`], [36.7 GB], [17.0 GB],
)

- The flux-differencing time step is now faster than the weak-form one (197 ms). The rest of the
  `rhs!` (surface terms, layout conversions) is now the larger part.
- Price: TPU-only code, and the two-point flux written again in JAX.

#small[*Pallas*: JAX's language for TPU kernels; Reactant embeds the lowered kernel with `Ops.hlo_call`.]

== E23: source code — Trixi's `HalfSweep` vs Pallas `kernel_slab`

#grid(
  columns: (1fr, 1.05fr),
  gutter: 0.8em,
  [
    #text(size: 13.5pt, weight: "bold")[Trixi `HalfSweep` (`dg_3d_gpu.jl`): 1 thread / node]
    #text(size: 10.5pt)[
      ```julia
      i, j, k, element = @index(Global, NTuple)
      flux_local = @localmem eltype(du) (NVARS, N, N, N)
      u_node = get_node_vars(u, equations, dg, i, j, k, element)
      @unroll for offset in 1:div(N, 2)   # half of the pairs
        ii = mod(i - 1 + offset, N) + 1
        u_ii = get_node_vars(u, equations, dg, ii, j, k, element)
        Ja_avg = 0.5f0 * (Ja1_node + Ja1_node_ii)
        flux = volume_flux(u_node, u_ii, Ja_avg, equations)
        flux_local[:, i, j, k] = flux     # stage in shared mem
        @synchronize                      # barrier 1
        iib = mod(i - 1 - offset, N) + 1
        du_local += (w * D_split[i, ii])  * flux +
                    (w * D_split[i, iib]) * flux_local[:, iib, j, k]
        @synchronize                      # barrier 2
      end
      ```
    ]
  ],
  [
    #text(size: 13.5pt, weight: "bold")[Pallas `kernel_slab` (`e23_pallas_fd.py`): $B$ elements / block]
    #text(size: 10.5pt)[
      ```python
      # DMA block (c, N, N, N, B) from HBM -> VMEM; D_split in SMEM
      blk = lambda c: pl.BlockSpec((c, N, N, N, B), lambda e: (0,0,0,0,e))
      pl.pallas_call(kernel_slab, grid=(E // B,),
        in_specs=[pl.BlockSpec(memory_space=SMEM), blk(5), blk(9)],
        out_specs=blk(5))

      def kernel_slab(ds_ref, u_ref, ja_ref, o_ref):
        a = prim([u_ref[v] for v in range(5)])  # once in VMEM (as FluxTurbo)
        for d in range(3):
          for i in range(N):
            ai, jai = node_slice(a, d, i), node_slice(ja[d], d, i)
            for ii in range(N):
              if ii == i: continue              # skip zero diagonal only
              b = node_slice(a, d, ii)          # full sweep, no symmetry
              nrm = 0.5 * (jai + node_slice(ja[d], d, ii))
              acc += ds_ref[i, ii] * kg(ai, b, *nrm)
            o_ref[:, slab(d, i)] += acc         # VMEM accumulate, 0 barriers
      ```
    ]
  ],
)

#[
  #set text(size: 14.5pt)
  - *1. Does Pallas compute only half the pairs?* *No* — it is a `FullSweep + FluxTurbo` (skips the
    zero diagonal `ii == i`, 12 of 16 pairs, and precomputes `prim(u)` once in VMEM). At 2.19 ms
    (570 GB/s) it is already HBM-bandwidth-bound, so extra on-chip flops are free.
  - *2. Does it use shared memory?* *Yes — TPU `VMEM` + `SMEM`*, with *no `@synchronize` barriers*
    because one SIMD instruction stream processes the whole block of $B = 512$–$1024$ elements sequentially.
]

== E23: GPU vs TPU memory hierarchy and sweep comparison

#[
  #show table: set text(size: 12pt)
  #set table(inset: (x: 5pt, y: 3.5pt))
  #table(
    columns: (1.1fr, 1.25fr, 1.95fr),
    [*GPU (`HalfSweep` / `FullSweep`)*], [*TPU (`Pallas kernel_slab`)*], [*Role and architectural difference*],
    [Global memory (VRAM)], [*HBM* (16 GB on v5e)], [Full arrays `u`, `Ja`, `V` ($E = 262,144$ elements); 1 read + 1 write (≈ 570 GB/s)],
    [Shared memory (`@localmem`) + regs], [*VMEM* (Vector Memory, 16–32 MB)], [Holds `u_ref`, `ja_ref`, `o_ref`, `prim(u)` for *$B = 512$–$1024$ elements at once* (vs 1 el. / 48 KiB)],
    [Constant memory / scalar regs], [*SMEM* (Scalar Memory)], [Holds the $4 times 4$ matrix `D_split` (`ds_ref[i, ii]` scalar coefficients)],
    [SIMT: $N^3 = 64$ threads / element → *needs `@synchronize`*], [Wide SIMD (VPU, $8 times 128$ tiles) → *0 barriers*], [One instruction stream sweeps slabs `(N, N, B)` in VMEM; no inter-thread race exists],
  )

  #v(0.15em)

  #table(
    columns: (1.55fr, auto, auto, auto, auto, auto),
    [*3D FD volume kernel (262k el., $n = 4$)*], [*Flux pairs / dir.*], [*On-chip mem.*], [*Barriers*], [*HBM traffic*], [*Time*],
    [A100, Trixi `HalfSweep` (CUDA)], [$N(N-1)\/2 = 6$], [`flux_local` (shared)], [$6 floor(N\/2) = 12$], [≈ 1×], [*2.11 ms*],
    [A100, Trixi `FullSweep` (CUDA)], [$N^2 = 16$], [`u_local` (shared)], [1], [≈ 1×], [3.12 ms],
    [A100, Trixi `FullSweepGlobal` (CUDA)], [$N^2 = 16$], [none (L1/L2 cache)], [0], [cached], [3.95 ms],
    [TPU v5e, `FullSweepGlobal` *raised* (E11)], [$N^2 = 16$ (unrolled)], [XLA auto (spills)], [0], [≈ 100 arrays (34 GB)], [47.8 ms],
    [TPU v5e, *Pallas `kernel_slab`* (E23)], [$N(N-1) = 12$], [*VMEM + SMEM*], [*0*], [*1 read + 1 write*], [*2.19 ms*],
  )
]

== E23: why Pallas goes through Python (JAX), and how it embeds in Julia

#[
  #set text(size: 15.5pt)
  + *No low-level TPU kernel DSL in Julia yet*:
    - On GPUs, `CUDA.jl` / `KernelAbstractions.jl` compile Julia `@kernel` code with `@localmem` straight to PTX.
    - On TPUs, on-chip memory (`VMEM` / `SMEM`) and DMA are controlled by Google's closed-source Mosaic compiler inside `libtpu` (MLIR `tpu` dialect). The only public frontend that emits Mosaic `tpu_custom_call` kernels today is *JAX's Pallas* (`jax.experimental.pallas`).
  + *Reactant emits standard StableHLO, not Mosaic kernels*:
    - JAX lowers the Pallas kernel once to a StableHLO module containing `stablehlo.custom_call @tpu_custom_call`, and Reactant splices that MLIR into the Julia `rhs!` with `Reactant.Ops.hlo_call`.
  + *Zero Python overhead at run time*:
    - Python/JAX runs *only once at lowering time* (or offline to a `.mlir` file); during time stepping the whole RK stage runs natively inside XLA/Julia.
]

#grid(
  columns: (1fr, 1fr),
  gutter: 0.8em,
  [
    #text(size: 12.5pt, weight: "bold")[1. Lower once in JAX (`e23_pallas_fd.py`)]
    #text(size: 10.5pt)[
      ```python
      f = jax.jit(lambda ds, u, ja:
        fd_pallas(ds, u, ja, B, "slab"))
      mlir = f.trace(
        spec(N, N), spec(5, N, N, N, E), spec(9, N, N, N, E)
      ).lower(lowering_platforms=("tpu",)).as_text()
      ```
    ]
  ],
  [
    #text(size: 12.5pt, weight: "bold")[2. Embed in Julia `rhs!` (`e12_p4est_array.jl`)]
    #text(size: 10.5pt)[
      ```julia
      function volume_pallas(p, cfg, U)
        us = cat(map(lead, U)...; dims = 1)
        V  = Reactant.Ops.hlo_call(
               PALLAS_MLIR, p.dsplit, us, p.Jall)[1]
        return ntuple(c -> reshape(V[c:c,:,:,:,:], n,n,n,E), 5)
      end
      ```
    ]
  ],
)

== E26: the whole RK stage in one Pallas kernel

E23 left ≈ 30 ms outside the volume term. E26 does everything in one kernel, on the brick path's
state as it is: volume term, surface fluxes (neighbours from the same block or one extra node layer),
Dirichlet values, source terms, Jacobian — and the RK stage update.

#[
  #show table: set text(size: 13.5pt)
  #table(
    columns: (2.2fr, auto, auto),
    [*262,144 elements, flux differencing, Float32*], [*`rhs!`*], [*RK step*],
    [TPU v5e, XLA (E12, E14)], [55.6 ms], [283.8 ms],
    [TPU v5e, Pallas volume kernel (E23)], [32.7 ms], [165.4 ms],
    [TPU v5e, whole `rhs!` in one kernel], [3.1 ms#super[a]], [28.7 ms],
    [TPU v5e, *whole RK stage in one kernel*], [—], [*18.2 ms*],
    [A100, Trixi's CUDA kernels (E17)], [6.5 ms], [≈ 33 ms#super[b]],
  )
]

#[
  #set text(size: 16pt)
  - Correct: the whole run (317 steps) is within the GPU test's Float32 tolerances. 3.6 ms per stage
    ≈ 650 GB/s, about 80% of the chip's peak: only the necessary data moves.
  - The time loop has to be one compiled program: at every program boundary XLA copies each kernel
    input into the kernel's layout (≈ 5 ms; one program per stage: 51.8 ms per step).
]

#small[
  #super[a] Kernel alone; as a program of its own 8.0 ms (the copies). #super[b] 5 × `rhs!`, without the
  update. RK step: one program per stage for XLA and E23 (their best, E14), one for all steps for E26.
]

== E24: several TPU chips

The arrays are split along the last element dimension over the chips (Reactant sharding
annotations); XLA inserts the communication between neighbouring chips. No other code changes.

#table(
  columns: (auto, auto, auto, auto, auto),
  [*Elements*], [*Volume term*], [*1 chip*], [*2 chips*], [*4 chips*],
  table.cell(rowspan: 2)[262,144], [weak form], [38.9 ms], [20.4 ms], [*9.69 ms* (4.0×)],
  [flux diff.], [55.7 ms], [26.5 ms], [*14.1 ms* (3.96×)],
  table.cell(rowspan: 2)[1.05 M], [weak form], [204 ms], [—], [42.6 ms],
  [flux diff.], [does not compile], [—], [62.0 ms],
  [2.1 M], [weak / flux diff.], [—], [—], [89.4 / 132.6 ms],
)

- Bitwise the same results as on one chip; linear scaling. About 500,000 elements fit per chip
  (16 GB). 8 chips: not possible within the project's quota (4 v5e chips per zone).

#small[TPU v5e, Float32, brick path, one `rhs!`.]

= Upstream issues

== Issue reports (drafted with minimal examples, not posted yet)

#[
  #set text(size: 17pt)
  - *KernelAbstractions*: the bounds check of a 4D index range cannot be raised (E11).
  - *Reactant*: converting `UInt8` to a float in a raised kernel (E13); `reshape` inside a kernel
    calls an undefined `_derived_array` (E13); printing a method signature with an
    `SVector` type parameter during tracing (E12); the `dot` / `norm` / `map` overlays treat static
    vectors of traced numbers as XLA arrays (E12, E25); arithmetic on traced `Bool`s (E25).
  - *Trixi*: the KA (GPU) path has no mortars (E18); patterns that block tracing — branches on values,
    equation fields restricted to `Real`, methods defined in elixirs and world age (E15).
  - *Trixi PR \#3332*: the `UInt8` enum base type meets the Reactant bug above; `Int8` works (E13).
]

#small[
  All are worked around in the prototype; none requires changing Trixi to run it. Re-tested with
  Reactant 0.2.297 (22 Oct): the E11 compiler crash is gone — Trixi's FullSweepGlobal now raises
  unchanged, without gathers; all others are still there.
]

= Summary

== Assessment update

#[
  #set text(size: 16.5pt)
  - *Coverage*: the array route reproduces Trixi's P4est solver for hyperbolic problems in 2D and 3D
    — curved unstructured meshes, mortars, AMR without recompiling, shock capturing, MHD — to
    round-off in Float64, with Trixi's physics functions unchanged; whole simulations agree within
    the Float32 tolerances. (Upstream Trixi's GPU path cannot run non-conforming meshes.)
  - *Speed, one TPU v5e chip vs Trixi on an A100* (3D, 262,144 elements): weak form 38.9 vs
    36.0 ms per `rhs!`; flux differencing with each RK stage in one Pallas kernel 18.2 vs ≈ 33 ms per
    RK step (E26). 2D (16.8 M nodes): 18.3 / 27.2 vs 16.4 / 13.7 ms per `rhs!`.
  - *Several chips* scale linearly: 4 chips 4× faster, ≈ 500,000 elements per chip.
  - *Data movement was the limit*: XLA's code moves about 10× the necessary data; one hand-written
    kernel per RK stage moves only the necessary data (≈ 80% of peak), at the price of TPU-only code
    with the physics written again in JAX.
  - *My reading*: Trixi can run on TPUs with the full P4est feature set. Per chip, the weak form
    in XLA is on par with Trixi's GPU kernels; flux differencing passes them only with hand-written
    kernels (so far Euler 3D on brick meshes). For Trixi in general GPUs remain the simpler target;
    TPUs pay off where a few kernels are worth writing.
]

= Next steps

== Plan and open questions

#[
  #set text(size: 17pt)
  + *Post the issue reports* (drafts ready).
  + *A fair GPU comparison*: the same fused kernel on an A100 (next slide).
  + *Speed*: the fused kernel (E26) for more equations and for unstructured meshes; AMR's host
    part (indicator, projection) on the device.
  + *Package extension*: the prototype as a Reactant backend for Trixi's P4est solver.
  + *AD*: not started (via Trixi Slack, as discussed).
]

*Open questions*

#[
  #set text(size: 17pt)
  - Is TPU-only kernel code (Pallas) acceptable in a package extension?
  - Are more chips worth a quota request?
  - AD: you mentioned Valentin Churavy's AD pull requests for Trixi. Are they Enzyme.jl on Trixi's
    native path? (My guess, not checked — I would ask him on Trixi Slack.)
]

== Next: the same fused kernel on an A100

*Why*: E26 compares a hand-fused TPU kernel with Trixi's current, unfused GPU kernels. An A100 has
2.5× the memory bandwidth of a v5e (≈ 2 TB/s vs 819 GB/s), so a GPU kernel fused the same way
should be faster per chip, and per euro the two may be close. Only the same optimization on both
sides shows whether the TPU hardware itself pays off.

*How*:

#[
  #set text(size: 17pt)
  - Start from Trixi's `HalfSweep` kernel (volume term in shared memory) and add the surface
    integral, boundary values, source terms, the Jacobian and even the RK stage update to the same
    kernel; intermediate values stay in registers and shared memory, nothing goes back to device
    memory in between.
  - Read the neighbours' face values straight from global memory: on a GPU such indexed reads
    (gathers) are much cheaper than on a TPU (E6).
  - Run it with CUDA.jl on an A100; compare time and achieved bandwidth with E26's TPU kernel.
]

#small[
  If bandwidth-bound like E26: ≈ 2.4 GB per stage at ≈ 1.5 TB/s → ≈ 1.6 ms per stage, ≈ 8 ms per RK
  step (v5e: 18.2 ms). On-demand prices: A100 ≈ \$1.2–1.8/h (RunPod), v5e €1.37/h (GCP).
]
