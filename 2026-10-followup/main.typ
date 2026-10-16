#import "@preview/touying:0.6.1": *
#import themes.metropolis: *

#show: metropolis-theme.with(
  aspect-ratio: "16-9",
  footer: self => self.info.title,
  config-info(
    title: [Trixi.jl on TPUs via Reactant.jl],
    subtitle: [Follow-up: P4est array prototype, kernel raising, enums],
    author: [Junyi Xu],
    date: datetime(year: 2026, month: 10, day: 16),
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
  [*E12*: the DG `rhs!` as array operations for 3D `P4estMesh`, with Trixi's own pointwise physics functions],
  [#ok],
  [`P4estMesh` in 2D and 3D, 3D more important],
  [3D: curved, unstructured, conforming meshes. 2D: not yet],
  [#warn],
  [Float32; test cases from Trixi's GPU tests],
  [`elixir_euler_source_terms_nonperiodic.jl` (3D GPU test) and its curved HOHQMesh variant],
  [#ok],
  [one of HalfSweep / FullSweep / FullSweepGlobal through Reactant],
  [*E11*: FullSweepGlobal raises and is correct on a TPU],
  [#ok],
  [`Symbol` → enum (Trixi PR \#3332)],
  [*E13*: the PR tested with the kernel inventory (E4) and the whole `rhs!` (E3)],
  [#ok],
)

#small[
  *Conforming* mesh: every interior face is shared by exactly two whole elements — no hanging nodes,
  so no *mortars* (Trixi's coupling across faces between elements of different size). \
  Hardware: one TPU v5e chip; for E11 also an NVIDIA L4 GPU. Float32 unless noted.
]

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
    five separate `rhs!` calls. Cause not found.
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

= Summary

== Assessment update

#[
  #set text(size: 17pt)
  - *Raising works for a 3D volume kernel*: FullSweepGlobal raises and is correct on a TPU after
    small changes — but it is slow there (47.8 ms at 16.8 M nodes; Trixi's whole `rhs!` on an A100:
    5.2 ms).
  - *The array route covers 3D `P4estMesh`* (curved, unstructured, conforming) and reproduces
    Trixi's `rhs!` to round-off, reusing Trixi's physics functions unchanged.
  - *Speed*: 2.3 ns per node and `rhs!` on brick meshes, ≈ 10 ns on curved unstructured meshes
    (weak form). Flux differencing stays 11× behind Trixi's kernel on an A100.
  - *Data movement is the limit*, for raised and array code alike: ≈ 100 state-sized reads/writes
    per `rhs!` where ≈ 2.5 would do. The one big lever found was the memory layout.
  - *My reading* is unchanged: for Trixi in general, GPUs remain the better target. In 3D a TPU
    would need far less data movement per `rhs!` than XLA produces from either form — e.g. from
    hand-written TPU kernels.
]

= Next steps

== Plan and open question

#[
  #set text(size: 17pt)
  + *Report the issues* with minimal examples: KA (4D bounds check), Reactant (compiler crash,
    `UInt8` kernel arguments, the E3 bug), Trixi PR \#3332 (`UInt8` base type).
  + *2D `P4estMesh`* in the prototype (3D only so far).
  + *Mortars* (non-conforming meshes, AMR): not covered by any experiment yet.
  + *Less data movement*: tree faces without the copy; why flux differencing compiles so slowly
    in the time loop; a hand-written TPU kernel for the volume term (Pallas — untried).
  + *Package extension* once the prototype has settled.
]

*Open question*: coverage first (steps 2–3) or speed first (step 4)? Hand-written kernels are the
remaining way to close the 3D gap, but they are TPU-only code.

#small[
  *Pallas*: JAX's language for hand-written TPU kernels. Reactant can call JAX functions through
  its Python interface and embed the compiled result, so a Pallas kernel could in principle sit
  inside a Reactant program.
]
