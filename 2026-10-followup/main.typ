#import "@preview/touying:0.6.1": *
#import themes.metropolis: *

#show: metropolis-theme.with(
  aspect-ratio: "16-9",
  footer: self => self.info.title,
  config-info(
    title: [Trixi.jl on TPUs via Reactant.jl],
    subtitle: [Follow-up: array prototype, Pallas kernels, comparison with Trixi],
    author: [Junyi Xu],
    date: datetime(year: 2026, month: 10, day: 23),
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
// issue in the experiment repository
#let repo-issue(n) = link("https://github.com/junyixu/TrixiReactant.jl/issues/" + str(n))[TrixiReactant.jl \##n]

#title-slide()

= Outline <touying:hidden>

#outline(title: none, indent: 1em, depth: 1)


= Since the last meeting

== Progress since 13 October

#[
  #show table: set text(size: 13pt)
  #table(
    columns: (1.15fr, 2.6fr, auto),
    [*Since 13 October*], [*What I did → result, compared with Trixi*], [*Continue?*],
    [array implementation, first as a separate prototype],
    [*Array code*: `rhs!` written as whole-array operations on `P4estMesh`, calling Trixi's
      physics functions unchanged; agrees with Trixi's `rhs!` to round-off],
    [#ok yes],
    [`P4estMesh` 2D and 3D; Float32; Trixi's GPU tests],
    [Every case of Trixi's GPU tests runs to the end on the TPU within the test tolerances; also
      mortars and AMR, which Trixi's GPU path does not have],
    [#ok done],
    [new: speed on the TPU],
    [one v5e vs Trixi on an A100 (3D, 262,144 elements): weak form 38.9 vs 36.0 ms per `rhs!`;
      flux differencing with hand-written TPU kernels (Pallas) 18.2 vs ≈ 33 ms per RK step],
    [#ok yes],
    [one of Trixi's flux-differencing kernels through Reactant],
    [Trixi's FullSweepGlobal kernel with its node loops unrolled (so that Reactant emits no
      gathers): correct on the TPU, but 47.8 ms; Trixi's HalfSweep on an A100 2.11 ms],
    [#no no],
    [`Symbol` → enum (Trixi PR \#3332)],
    [With the PR, Trixi's own kernels raise without my integer workarounds (`Int8`; `UInt8` hits a
      Reactant bug: #repo-issue(6)). The array code does not need it. Why `Symbol`s fail in XLA,
      tests with the PR: #repo-issue(9)],
    [#ok tested],
    [Reactant first: it will expose low-level problems],
    [3 issues reported to Reactant.jl, fixes for 2 of them as pull requests],
    [#ok as found],
    [question: the same speed by changing KA kernels, without Reactant?],
    [A100, no Reactant: the unrolled loops above make FullSweepGlobal 7–9× slower (26–29 vs
      3.3–4.1 ms); they only help raising. A fused KA kernel like the Pallas one: not tested yet],
    [#warn next step 1],
  )
]

#small[
  *Mortars*: Trixi's coupling across faces between elements of different size (non-conforming
  meshes, AMR). *Pallas*: JAX's language for hand-written TPU kernels. Float32 unless noted.
]

= Background: slices and gathers

== What a gather is

Reading an array at positions that are themselves *data*, stored in an integer array.

// one row of the array picture: the cells at the positions in `hl` are filled
#let mem-row(label, hl, fill, result) = (
  align(right, label),
  ..(10, 20, 30, 40, 50, 60).enumerate().map(((i, v)) => table.cell(
    stroke: 0.5pt + luma(150),
    fill: if (i + 1) in hl { fill },
  )[#v]),
  align(left, result),
)

#grid(
  columns: (1fr, 1.15fr),
  gutter: 1em,
  align: top,
  [
    #text(size: 18pt)[
      ```julia
      x = [10, 20, 30, 40, 50, 60]

      # slice: positions fixed in the program
      x[2:4]          # [20, 30, 40]

      # gather: positions read from an array
      idx = [6, 1, 4]
      x[idx]          # [60, 10, 40]

      # the same as
      [x[idx[k]] for k in 1:3]
      ```
    ]
  ],
  [
    #table(
      columns: (auto,) + (2em,) * 6 + (auto,),
      align: center + horizon,
      stroke: none,
      inset: (x: 4pt, y: 5pt),
      [], ..range(1, 7).map(i => text(size: 11pt, fill: luma(110), str(i))), [],
      ..mem-row(`x[2:4]`, (2, 3, 4), rgb("#bbdefb"), [→ `[20, 30, 40]`]),
      ..mem-row(`x[idx]`, (6, 1, 4), rgb("#ffe0b2"), [→ `[60, 10, 40]`]),
      align(right, text(size: 13pt, fill: rgb("#ef6c00"))[`k`]),
      ..("2", "", "", "3", "", "1").map(s => text(size: 13pt, fill: rgb("#ef6c00"), s)), [],
    )
    #set text(size: 16pt)
    - *Slice*: one block whose start and length the compiler knows → copied at full memory
      bandwidth.
    - *Gather*: the positions are known only at run time, after `idx` is read → every index is
      a separate lookup.
  ]
)

#[
  #set text(size: 17pt)
  In a DG solver: the left neighbour of element `e` is `e - 1` on a brick mesh (*slice*), but
  `left[e]`, read from the mesh, on an unstructured mesh (*gather*).
]

== Gathers on the TPU: the cost is per index

`X` has one column per block of $M$ numbers (e.g. one face: all variables at its nodes). How many
numbers one index fetches depends on how the code is written:

#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  align: top,
  [
    #text(size: 13.5pt, weight: "bold")[Raised kernel: one thread per number]
    #text(size: 17pt)[
      ```julia
      i, k = @index(Global, NTuple)
      Y[i, k] = X[i, idx[k]]   # 1 index → 1 number
      ```
    ]
  ],
  [
    #text(size: 13.5pt, weight: "bold")[Array code: one index per column]
    #text(size: 17pt)[
      ```julia
      Y = X[:, idx]            # 1 index → M numbers
      ```
    ]
  ],
)

Measured: 64 MiB of Float32 read through `idx`, GB/s (plain copy: 365):

#table(
  columns: (2.2fr, auto, auto, auto),
  [*numbers per index $M$*], [$1$], [$16$], [$256$ (1 KiB)],
  [TPU v5e, slice (structured neighbour)], [164], [375], [360],
  [TPU v5e, gather `X[:, idx]`], [1.0], [19], [*234*],
  [A100, gather `X[:, idx]` (random `idx`)], [130], [577], [671],
)

#[
  #set text(size: 16pt)
  - The TPU looks up ≈ 120 M indices per second (≈ 8 ns each), whatever $M$, random or regular
    alike: only ≈ 1 KiB per index comes close to a copy.
  - Raised kernels gather one number per index. The array code gathers one whole face per
    index (3D Euler, $n = 4$: 80 numbers). Brick meshes need no gather at all.
  - On an A100 even single-number gathers are fast: the per-index cost is a TPU property.
]


= Trixi's own kernel through Reactant

== Trixi's flux-differencing kernel on the TPU

Trixi's `FullSweepGlobal` (3D, one thread per node) raised by Reactant; the node loops (simplified):

#grid(
  columns: (1fr, 1fr),
  gutter: 0.8em,
  align: top,
  [
    #text(size: 13.5pt, weight: "bold")[Trixi: loop over the nodes of a line]
    #text(size: 12.5pt)[
      ```julia
      for ii in 1:n
        u_ii = get_node_vars(u, equations, dg,
                             ii, j, k, element)
        du += D_split[i, ii] *
              volume_flux(u_node, u_ii, Ja_avg, equations)
      end
      ```
    ]
    #text(size: 14pt)[Raised: `ii` is a loop variable, so `u[:, ii, j, k, element]` is a gather —
      244 gathers.]
  ],
  [
    #text(size: 13.5pt, weight: "bold")[Changed: loop unrolled ($n$ known when compiling)]
    #text(size: 12.5pt)[
      ```julia
      du = unrolled_sum(Val(n)) do ii
        u_ii = get_node_vars(u, equations, dg,
                             ii, j, k, element)
        D_split[i, ii] *
          volume_flux(u_node, u_ii, Ja_avg, equations)
      end
      ```
    ]
    #text(size: 14pt)[Raised: `ii` is a constant, so plain slices — 0 gathers.]
  ],
)

#table(
  columns: (2.2fr, auto),
  [*Flux-differencing volume term, 262,144 elements, Float32*], [*time*],
  [TPU v5e, Trixi's kernel unrolled and raised], [47.8 ms],
  [A100, Trixi's HalfSweep, CUDA], [*2.11 ms*],
  [TPU v5e, Pallas kernel (below)], [2.19 ms],
)

#[
  #set text(size: 15.5pt)
  - Correct on the TPU (≤ $3.5 times 10^(-7)$ from Trixi's kernel), but 23× slower than Trixi on an
    A100 (XLA: ≈ 240 kernels, ≈ 100 state-sized reads and writes). Not worth continuing for speed.
]

#small[Also needed: `@kernel unsafe_indices = true` (KA's 4D bounds check does not raise) and a fixed
  workgroup size.]

= Array prototype

== Array code: Trixi's functions applied to whole arrays

Any conforming 3D or 2D `P4estMesh` set up by Trixi; one $(n, n, n, "elements")$ array per variable.
Weak-form volume term (simplified):

#grid(
  columns: (1fr, 1fr),
  gutter: 0.8em,
  align: top,
  [
    #text(size: 13.5pt, weight: "bold")[Trixi (`dg_3d.jl`): loops, on a GPU one thread / element]
    #text(size: 11pt)[
      ```julia
      for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
        u_node = get_node_vars(u, equations, dg, i, j, k, element)
        flux1 = flux(u_node, 1, equations)
        # ... flux2, flux3
        contravariant_flux1 = Ja11 * flux1 + Ja12 * flux2 + Ja13 * flux3
        for ii in eachnode(dg)
          multiply_add_to_node_vars!(du, derivative_hat[ii, i],
            contravariant_flux1, equations, dg, ii, j, k, element)
        end
        # ... directions 2 and 3
      end
      ```
    ]
  ],
  [
    #text(size: 13.5pt, weight: "bold")[Array code: whole arrays]
    #text(size: 11pt)[
      ```julia
      # U: 5 arrays (n, n, n, elements)
      # Trixi's `flux`, applied to every node at once
      f = ntuple(o -> pointwise(u -> flux(u, o, equations),
                                U...), 3)
      for d in 1:3
        Fd = contract(Ja[d], f)             # Ja^d · f
        du = du .+ apply_dim(D_hat, Fd, d)  # matrix product
      end                                   # along node axis d
      ```
    ]
  ],
)

#table(
  columns: (2.4fr, auto, auto),
  [*`rhs!` vs Trixi, Euler 3D, rel. max. difference*], [*Float64*], [*Float32*],
  [Cartesian, 64 elements (GPU test)], [$1.3 times 10^(-13)$], [$5.6 times 10^(-5)$],
  [curved unstructured (HOHQMesh), 204 elements], [$2.1 times 10^(-13)$], [$1.0 times 10^(-4)$],
)

#[
  #set text(size: 15.5pt)
  - The same discretization as Trixi, to round-off, also on the TPU; no Trixi code changed. Other
    equations need no new flux code (MHD). Neighbours: slices on brick meshes, one gather of
    whole faces on any other mesh.
  - Worth continuing: this is the code for a package extension; it runs on GPUs too (next section).
]

== What the prototype covers, compared with Trixi

#table(
  columns: (1.3fr, 2.4fr, 1fr),
  [*Feature*], [*Prototype: result vs Trixi*], [*Trixi's GPU path*],
  [Trixi's GPU test cases, 3D and 2D: advection, Euler, MHD],
  [run to the end on the TPU within the tests' Float32 tolerances, same number of steps],
  [tested],
  [GLM-MHD, non-conservative terms],
  [Trixi's functions unchanged; `rhs!` $7.6 times 10^(-14)$ (Float64)],
  [tested],
  [curved unstructured meshes],
  [`rhs!` ≤ $9 times 10^(-13)$ in 3D and 2D],
  [not tested],
  [mortars (non-conforming meshes)],
  [`rhs!` ≤ $2.6 times 10^(-12)$ with 298 mortars],
  [not implemented],
  [AMR],
  [Trixi's `AMRCallback`; the same meshes as Trixi; 15–40 adaptations with 1–2 compilations],
  [not implemented (needs mortars)],
  [shock capturing],
  [`rhs!` $2.4 times 10^(-16)$; 10–13% more time than flux differencing],
  [not tested],
)

#[
  #set text(size: 16pt)
  - Float64 whole runs agree with Trixi to ≤ $10^(-12)$; Float32 runs on the TPU differ from Trixi
    as much as Trixi's own Float32 and Float64 runs.
  - Open: AMR adapts on the host, 0.03–1.3 s per adaptation vs 5–61 ms per RK step on the TPU.
    Worth moving to the device only if AMR runs matter.
]


= Speed

== Speed: array code, Pallas kernels and Trixi

3D Euler, 262,144 elements (16.8 M nodes), Float32:

#table(
  columns: (2.3fr, auto, auto, auto),
  [], [*weak form `rhs!`*], [*flux diff. `rhs!`*], [*flux diff. RK step*],
  [A100, Trixi's CUDA kernels], [36.0 ms], [6.5 ms], [≈ 33 ms#super[a]],
  [A100, array code (Reactant → CUDA)], [*12.9 ms*], [14.7 ms], [99 ms],
  [v5e, array code (XLA)], [38.9 ms], [55.7 ms], [283 ms],
  [v5e, + Pallas volume kernel], [—], [32.3 ms], [164 ms],
  [v5e, one Pallas kernel per RK stage], [—], [3.1 ms#super[b]], [*18.2 ms*],
)

#[
  #set text(size: 15.5pt)
  - *Weak form*: one v5e with XLA ≈ Trixi on an A100. On the A100 the same array code is 2.8× faster
    than Trixi, whose weak-form volume kernel runs one thread per element (78–85% of its `rhs!`).
  - *Flux differencing*: XLA is 8.6× slower than Trixi: each `rhs!` reads and writes ≈ 80–110
    state-sized arrays where ≈ 2.5 would do (hundreds of unfused small kernels, 150–190
    transposes). Pallas kernels move only the necessary data: one kernel per RK stage is 1.8×
    faster per step than Trixi on an A100.
  - Worth continuing: Pallas, yes — but Trixi's GPU kernels are not fused this way; the fair
    comparison is the same fusion on an A100 (next steps).
]

#small[
  #super[a] 5 × `rhs!`, without the RK update. #super[b] Kernel alone. RK step: one compiled
  program per stage; last row: one program for the whole time loop.
]

== Pallas volume kernel vs Trixi's HalfSweep

Volume term, 262,144 elements: HalfSweep (A100) 2.11 ms, Pallas (v5e) *2.19 ms*, XLA (v5e) 27.6 ms.

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
    #text(size: 13.5pt, weight: "bold")[Pallas `kernel_slab` (JAX): $B$ elements / block]
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
  - Nearly Trixi's time, on a chip with 2.5× less memory bandwidth (819 GB/s vs 2 TB/s). Pallas
    computes all pairs but the zero diagonal (like FullSweep): at ≈ 570 GB/s it is bandwidth-bound,
    so the extra arithmetic is free.
  - Trixi: one element per workgroup in shared memory, 12 barriers. Pallas: 512–1024 elements in
    VMEM, no barriers. Price: TPU-only code, the two-point flux written again in JAX.
]

== The whole RK stage in one Pallas kernel

#grid(
  columns: (1fr, 1.1fr),
  gutter: 0.8em,
  align: top,
  [
    #text(size: 13.5pt, weight: "bold")[Trixi `rhs_hyperbolic!` (`dg_2d_gpu.jl`, also 3D)]
    #text(size: 10.5pt)[
      ```julia
      # each call is a kernel: it reads its inputs from
      # and writes its results to device memory
      calc_volume_integral!(backend, du, u, ...)
      prolong2interfaces_and_calc_interface_flux!(
          backend, surface_flux_values, u, ...)
      prolong2boundaries!(backend, cache, u, ...)
      calc_boundary_flux!(backend, cache, t, ...)
      calc_surface_integral_and_apply_jacobian_and_calc_sources!(
          backend, du, u, t, ...)
      # then the RK stage update (OrdinaryDiffEq):
      # one more pass over du and u
      ```
    ]
  ],
  [
    #text(size: 13.5pt, weight: "bold")[One Pallas kernel: a block of elements in VMEM]
    #text(size: 10.5pt)[
      ```python
      def kernel(ds_ref, t_ref, *refs):
          u = [r[...] for r in u_r]              # read once
          volume_into(o_r, ds_ref, prim(u), ja)  # previous slide
          for d in range(3):
              for side in (0, 1):                # 6 faces
                  own = [layer(q, d, li) for q in u]
                  nbr = ...   # same block (roll), next block
                              # (one node layer) or Dirichlet
                  f = llf(own, nbr, *nrm)        # surface flux
                  for c in range(5):
                      o_r[c][idx] += FACTOR * f[c]
          s = source_terms(*X, t)
          for c in range(5):                     # RK stage
              kn = a * k_r[c][...] + dt * (-ij * o_r[c][...] + s[c])
              o_r[c][...] = kn
              uo_r[c][...] = u[c] + b * kn       # write once
      ```
    ]
  ],
)

#[
  #set text(size: 15pt)
  - 18.2 ms per RK step on a v5e vs ≈ 33 ms for Trixi on an A100; 3.6 ms per stage ≈ 650 GB/s, 80%
    of the v5e's peak. Correct: the whole run (317 steps) passes the GPU test.
  - Limits: 3D Euler on brick meshes only; the whole time loop must be one compiled program (each
    program boundary costs ≈ 5 ms of layout copies).
  - Worth continuing if TPU-only code is acceptable; first the same fusion on an A100.
]

== Unstructured meshes: the gather costs on the TPU

#grid(
  columns: (1fr, 1fr),
  gutter: 0.8em,
  align: top,
  [
    #text(size: 13.5pt, weight: "bold")[Brick mesh: neighbours from slices]
    #text(size: 12pt)[
      ```julia
      # 1D: the left neighbour of e is e - 1 (periodic)
      uL = hcat(u[N:N, end:end], u[N:N, 1:(end - 1)])
      ```
    ]
  ],
  [
    #text(size: 13.5pt, weight: "bold")[Any mesh: one gather of whole faces]
    #text(size: 12pt)[
      ```julia
      # 3D: faces (n, n, 5, 6E)
      # p.src[s]: the face whose values face s needs
      G = reshape(faces, n * n * 5, :)[:, p.src]
      # then one of 8 orientations per face
      ```
    ]
  ],
)

#table(
  columns: (2.3fr, auto, auto, auto),
  [*Weak-form `rhs!`, Float32*], [*neighbours*], [*time*], [*per node*],
  [v5e, 3D brick, 262,144 elements], [slices], [38.9 ms], [2.3 ns],
  [v5e, 3D curved unstructured, 13,056 elements], [gather], [17.3 ms], [11 ns],
  [v5e, 2D curved unstructured, 1.6 M elements], [gather], [389 ms], [15 ns],
  [A100, 3D, 262,144 elements], [slices / gather], [12.9 / 14.8 ms], [0.8 / 0.9 ns],
)

#[
  #set text(size: 15pt)
  - On the TPU a gather costs per index (≈ 8 ns): unstructured meshes take ≈ 5× (3D) to 14× (2D)
    longer per node than brick meshes. On an A100 the gather costs 15%, and the array code is still
    2.4× faster than Trixi (14.8 vs 36.0 ms).
  - Tried: tree blocks (slices inside each tree, gathers between trees): correct, but faster than
    the gather only with $8^3$ elements per tree (1.4×). AMR needs the gather path anyway.
  - Worth continuing only if unstructured meshes on TPUs matter; for them a GPU is the simpler target.
]

= Upstream issues

== Issues reported to Reactant.jl

#let reactant(kind, n) = link("https://github.com/EnzymeAD/Reactant.jl/" + kind + "/" + str(n))[\##n]

Three issues with minimal examples (Reactant 0.2.297); pull requests fix two of them:

#table(
  columns: (auto, 1fr, auto, auto),
  [*Issue*], [*Problem*], [*Found with*], [*Fix*],
  reactant("issues", 3538),
  [`dot` / `norm` / `map` on static vectors (`SVector`) of traced numbers go to the XLA-array
    versions and fail],
  [Trixi's fluxes (`dot`, `norm`)], [PR #reactant("pull", 3541)],
  reactant("issues", 3539),
  [arithmetic on traced `Bool`s (`true - false`) does not compile],
  [Trixi's 2D advection test], [PR #reactant("pull", 3540)],
  reactant("issues", 3553),
  [printing a method signature with an `SVector` type parameter fails during tracing],
  [advection velocity as a type parameter], [—],
)

#[
  #set text(size: 15pt)
  - *KernelAbstractions*: the bounds check of a 4D index range (only in Trixi's `FullSweepGlobal`)
    cannot be raised; worked around by adding an unrolled `in` method. Probably fixed in KA 0.10 by
    #link("https://github.com/JuliaGPU/KernelAbstractions.jl/pull/844")[\#844] (V. Churavy, 10 Oct:
    branch-free `reduce(&, ...)`); not tested, as Reactant and Trixi still require KA 0.9.
  - *Not reported yet*: most problems of Trixi's own kernels through Reactant (the issues above
    come from the array code): type-valued kernel arguments, NaNs from the raised 2D interface
    kernel, code that does not raise (loops, local arrays, `sinpi`, barriers); still in 0.2.297,
    with minimal examples: `UInt8` → float, `reshape` inside a kernel.
]

#small[The reported issues and the KA bounds check have workarounds that do not change Trixi.]


= Summary and next steps

== Assessment: what is worth continuing

#[
  #show table: set text(size: 14pt)
  #table(
    columns: (1.25fr, 2.6fr, 1.05fr),
    [*Direction*], [*Result, compared with Trixi*], [*Continue?*],
    [array code (XLA)],
    [the P4est features tested (curved meshes, mortars, AMR, shock capturing) agree with Trixi to
      round-off; Trixi's GPU path has no mortars or AMR. One v5e vs Trixi on an A100: weak form on
      par, flux differencing 8.6× slower without Pallas],
    [#ok as a package extension],
    [Pallas kernels],
    [flux differencing: RK step 1.8× faster than Trixi on an A100; so far 3D Euler on brick
      meshes, TPU-only code],
    [#ok after the A100 comparison],
    [Trixi's own kernels through Reactant],
    [correct, but 23× slower than Trixi on an A100],
    [#no],
    [unstructured meshes on TPUs],
    [5–14× slower per node than brick meshes; on an A100 the gather costs 15%],
    [#warn only if needed],
  )
]

#[
  #set text(size: 15pt)
  *My reading*: Trixi's P4est solver can run on TPUs with the features tested (curved meshes,
  mortars, AMR, shock capturing). Per chip, XLA alone matches
  Trixi's GPU kernels for the weak form; flux differencing passes them only with hand-written
  kernels, which are TPU-only.

  *Why array code*: XLA fuses it. On TPUs, Trixi's thread-style kernels need rewriting anyway: most
  did not raise (Reactant 0.2.291), and the flux-differencing one, raised, takes 47.8 ms (Pallas:
  2.19 ms). So the array code is the frame and Pallas speeds up the hot parts. On GPUs it beats
  Trixi's one-thread-per-element kernel (weak form, 2.8×), but not HalfSweep (flux differencing,
  2.3× slower).
]

== Next steps

#[
  #set text(size: 15pt)
  + *The same fusion on an A100*: Trixi's HalfSweep plus surface terms, sources and the RK update
    in one CUDA kernel, compared with the fused Pallas kernel. An A100 has 2.5× the bandwidth of a v5e; if
    bandwidth-bound, ≈ 8 ms per RK step (v5e: 18.2 ms). Prices: v5e €1.37/h, A100 ≈ \$1.2–1.8/h.
    This decides whether TPUs pay off.
  + *Package extension*: the array code as a Reactant backend for Trixi's P4est solver.
  + *KA 0.10 with Reactant*: check that 4D kernels raise without the workaround (\#844).
  + *More Pallas kernels* (other equations, unstructured meshes): only if step 1 favours TPUs.
  + *Several TPU chips*: Reactant can split the arrays along the element dimension without code
    changes, so I expect near-linear scaling; needs more TPU quota.
  + *AD*: not started. Reactant's claimed advantage (not verified yet): Enzyme differentiates the whole traced tensor
    program (StableHLO), and XLA optimizes the derivative like any other program.
    With KA, Enzyme differentiates each kernel separately in LLVM IR (a tape per thread, atomic adds where threads
    read the same data).
]

== Open questions

#[
  #set text(size: 16pt)
  - Is TPU-only kernel code (Pallas) acceptable in a package extension?
  - *Code structure*: Trixi's KA kernels keep one structure (loops over elements) for CPUs,
    including MPI clusters, and GPUs. The array code is a second structure: whole-array operations
    that XLA fuses (reasons on the assessment slide). Is that worth maintaining in Trixi?
  - AD: find a potential application (maybe in November).
]
