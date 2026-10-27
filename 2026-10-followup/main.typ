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
#let repo-issue(n) = link("https://github.com/junyixu/TrixiReactant.jl/issues/" + str(n))[\##n]

#title-slide()

= Outline <touying:hidden>

#outline(title: none, indent: 1em, depth: 1)


= Since the last meeting

== Progress since 13 October (1/2)

#let progress-table(..rows) = [
  #show table: set text(size: 13pt)
  #table(
    columns: (1.15fr, 2.6fr, auto),
    [*From the 13 October meeting*], [*What I did and the result, compared with Trixi*],
    [*Continue?*],
    ..rows,
  )
]

#progress-table(
  [write an array implementation, first as a separate prototype],
  [I wrote `rhs!` as whole-array operations on `P4estMesh` (the *array code*). It calls Trixi's
    physics functions unchanged and agrees with Trixi's `rhs!` to round-off.],
  [#ok yes],
  [`P4estMesh` in 2D and 3D, Float32, the cases of Trixi's GPU tests],
  [Every case of Trixi's GPU tests runs to the end on the TPU within the test tolerances. Mortars
    and AMR work too, although Trixi's GPU path does not have them.],
  [#ok done],
  [new since the meeting: speed on the TPU],
  [With 262,144 elements in 3D, the array code on one TPU v5e takes 38.9 ms per weak-form `rhs!`;
    Trixi on an A100 takes 36.0 ms. For flux differencing with hand-written TPU kernels (Pallas),
    an RK step takes 18.2 ms, against about 33 ms for Trixi.],
  [#ok yes],
  [run one of Trixi's flux-differencing kernels through Reactant],
  [With its node loops unrolled, so that Reactant emits no gathers, Trixi's FullSweepGlobal kernel
    is correct on the TPU but takes 47.8 ms. Trixi's HalfSweep takes 2.11 ms on an A100.],
  [#no no],
)

#small[
  *Mortars*: Trixi's coupling across faces between elements of different size (non-conforming
  meshes, AMR). *Pallas*: JAX's language for hand-written TPU kernels. Float32 unless noted.
]

== Progress since 13 October (2/2)

#progress-table(
  [replace the `Symbol`s in `node_indices` by an enum (Trixi PR \#3332)],
  [With the PR, Trixi's own kernels raise without my integer workarounds. Only a small adapter
    remains, which stores the enum values as `Int8`, because `UInt8` hits a Reactant bug
    (#repo-issue(6)). The array code does not need the PR. #repo-issue(9) explains why `Symbol`s
    cannot go into XLA arrays and how I tested the PR.],
  [#ok tested],
  [get Reactant running first: it will expose low-level problems],
  [3 issues reported to Reactant.jl, fixes for 2 of them as pull requests],
  [#ok keep reporting],
  [your question: can changed KA kernels reach the same speed without Reactant?],
  [Unrolling the node loops lets Reactant raise FullSweepGlobal, but on an A100 without Reactant
    it makes the kernel 7–9× slower (26–29 ms instead of 3.3–4.1 ms). One KA kernel for the whole
    RK stage, like the Pallas kernel, is not written yet (#repo-issue(12), analysis in
    #repo-issue(14)).],
  [#warn next step 1],
)

#small[Numbers such as \#6 link to issues in my repository TrixiReactant.jl.]

= Background: slices and gathers

== What a gather is

A gather reads an array at positions that are themselves *data*, stored in an integer array.

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
    - *Slice*: one block whose start and length the compiler knows, so it is copied at full
      memory bandwidth.
    - *Gather*: the positions are known only at run time, after `idx` is read, so every index is
      a separate lookup.
  ]
)

#[
  #set text(size: 17pt)
  In a DG solver, the left neighbour of element `e` is `e - 1` on a brick mesh (a *slice*), but on
  an unstructured mesh it is `left[e]`, read from the mesh (a *gather*).
]

== Gathers on the TPU: the cost is per index

`X` has one column per block of $M$ numbers, for example all variables at the nodes of one face.
How many numbers one index fetches depends on how the code is written:

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

Bandwidth in GB/s for reading 64 MiB of Float32 through `idx` (a plain copy on the v5e: 365):

#table(
  columns: (2.2fr, auto, auto, auto),
  [*numbers per index $M$*], [$1$], [$16$], [$256$ (1 KiB)],
  [TPU v5e, slice (structured neighbour)], [164], [375], [360],
  [TPU v5e, gather `X[:, idx]`], [1.0], [19], [*234*],
  [A100, gather `X[:, idx]` (random `idx`)], [130], [577], [671],
)

#[
  #set text(size: 15pt)
  - The TPU looks up about 120 million indices per second (≈ 8 ns each), whatever $M$ is and
    whether the indices are random or regular. Only about 1 KiB per index comes close to a copy.
  - Raised kernels gather one number per index. The array code gathers one whole face per
    index (80 numbers for 3D Euler with $n = 4$). Brick meshes need no gather at all.
  - On an A100 even single-number gathers are fast, so the per-index cost is specific to the TPU.
]


= Trixi's own kernel through Reactant

== Trixi's flux-differencing kernel through Reactant

Trixi's `FullSweepGlobal` kernel (3D, one thread per node), raised by Reactant. The node loops, simplified:

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
    #text(size: 14pt)[When raised, `ii` is a loop variable, so `u[:, ii, j, k, element]` is a
      gather; the kernel has 244 gathers.]
  ],
  [
    #text(size: 13.5pt, weight: "bold")[My change: the loop unrolled ($n$ is known when compiling)]
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
    #text(size: 14pt)[When raised, `ii` is a constant, so there are no gathers.]
  ],
)

#small[For the TPU run the kernel also needed `@kernel unsafe_indices = true`, because KA's 4D bounds
  check does not raise, and a fixed workgroup size.]

== Trixi's flux-differencing kernel on the TPU

#table(
  columns: (2.2fr, auto),
  [*Flux-differencing volume term, 262,144 elements, Float32*], [*time*],
  [TPU v5e, Trixi's kernel unrolled and raised], [47.8 ms],
  [A100, Trixi's HalfSweep, CUDA], [*2.11 ms*],
  [TPU v5e, Pallas kernel (below)], [2.19 ms],
)

#[
  #set text(size: 16pt)
  - The raised kernel is correct on the TPU (at most $3.5 times 10^(-7)$ from Trixi's kernel), but
    23× slower than HalfSweep on an A100: XLA turns it into about 240 kernels, which together read
    and write about 100 arrays the size of the state. For speed it is not worth continuing.
]

= Array prototype

== Array code: Trixi's functions applied to whole arrays

The array code takes any conforming 3D or 2D `P4estMesh` set up by Trixi and stores one
$(n, n, n, "elements")$ array per variable. The weak-form volume term, simplified:

#grid(
  columns: (1fr, 1fr),
  gutter: 0.8em,
  align: top,
  [
    #text(size: 13.5pt, weight: "bold")[Trixi (`dg_3d.jl`): loops; on a GPU, one thread per element]
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
  [*Euler 3D: relative difference from Trixi's `rhs!`*], [*Float64*], [*Float32*],
  [Cartesian, 64 elements (GPU test)], [$1.3 times 10^(-13)$], [$5.6 times 10^(-5)$],
  [curved unstructured (HOHQMesh), 204 elements], [$2.1 times 10^(-13)$], [$1.0 times 10^(-4)$],
)

#[
  #set text(size: 15.5pt)
  - The array code computes the same discretization as Trixi, to round-off, also on the TPU, and
    no Trixi code was changed. Other equations need no new flux code (MHD is an example).
    Neighbours come from slices on brick meshes and from one gather of whole faces on other meshes.
  - It is worth continuing as the code for a package extension; it also runs on GPUs.
]

== What the prototype covers, compared with Trixi

#[
#show table: set text(size: 14pt)
#table(
  columns: (1.3fr, 2.4fr, 1fr),
  [*Feature*], [*Prototype: result compared with Trixi*], [*Trixi's GPU path*],
  [Trixi's GPU test cases, 3D and 2D: advection, Euler, MHD],
  [they run to the end on the TPU within the tests' Float32 tolerances, with as many steps as Trixi],
  [tested],
  [GLM-MHD, non-conservative terms],
  [uses Trixi's functions unchanged; `rhs!` differs by $7.6 times 10^(-14)$ (Float64)],
  [tested],
  [curved unstructured meshes],
  [`rhs!` differs by at most $9 times 10^(-13)$ in 3D and 2D],
  [not tested],
  [mortars (non-conforming meshes)],
  [`rhs!` differs by at most $2.6 times 10^(-12)$ with 298 mortars],
  [not implemented],
  [AMR],
  [adapts with Trixi's `AMRCallback` and gets the same meshes as Trixi; 15–40 adaptations need only 1–2 compilations],
  [not implemented (needs mortars)],
  [shock capturing],
  [`rhs!` differs by $2.4 times 10^(-16)$; it takes 10–13% more time than flux differencing alone],
  [not tested],
)
]

#[
  #set text(size: 15pt)
  - Whole runs in Float64 agree with Trixi to $10^(-12)$ or better. Float32 runs on the TPU differ
    from Trixi about as much as Trixi's own Float32 and Float64 runs differ from each other.
  - Still open: AMR adapts the mesh on the host, which takes 0.03–1.3 s per adaptation, while an RK
    step on the TPU takes 5–61 ms. Moving AMR to the device is worth it only if AMR runs matter.
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
  [v5e, array code + Pallas volume kernel], [—], [32.3 ms], [164 ms],
  [v5e, one Pallas kernel per RK stage], [—], [3.1 ms#super[b]], [*18.2 ms*],
)

#[
  #set text(size: 14.5pt)
  - *Weak form*: the array code on one v5e is about as fast as Trixi on an A100. On the A100 the
    same array code is 2.8× faster than Trixi, whose weak-form volume kernel runs one thread per
    element and takes 78–85% of its `rhs!`.
  - *Flux differencing*: the array code on the v5e is 8.6× slower than Trixi on the A100. Each
    `rhs!` reads and writes about 80–110 arrays the size of the state where about 2.5 would do,
    because XLA leaves hundreds of small kernels unfused and adds 150–190 transposes. Pallas kernels
    move only the data they need: with one kernel per RK stage, a step is 1.8× faster than Trixi.
  - Pallas is worth continuing. But Trixi's GPU kernels are not fused this way, so the fair
    comparison is the same fusion on an A100 (see the next steps).
]

#small[
  #super[a] Five `rhs!` calls, without the RK updates. #super[b] The kernel alone. RK steps use one
  compiled program per stage, except in the last row, which uses one program for the whole time loop.
]

== Pallas volume kernel vs Trixi's HalfSweep

Volume term for 262,144 elements: HalfSweep on an A100 2.11 ms, Pallas on a v5e *2.19 ms*, XLA on a
v5e 27.6 ms.

#grid(
  columns: (1fr, 1.05fr),
  gutter: 0.8em,
  [
    #text(size: 13.5pt, weight: "bold")[Trixi `HalfSweep` (`dg_3d_gpu.jl`): one thread per node]
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
    #text(size: 13.5pt, weight: "bold")[Pallas `kernel_slab` (JAX): $B$ elements per block]
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
  #set text(size: 14pt)
  - The Pallas kernel nearly matches Trixi's time on a chip with 2.5× less memory bandwidth
    (819 GB/s against 2 TB/s). It computes all node pairs except the zero diagonal, as FullSweep
    does; at about 570 GB/s it is bandwidth-bound, so the extra arithmetic costs nothing.
  - Trixi keeps one element per workgroup in shared memory and needs 12 barriers; the Pallas kernel
    keeps 512–1024 elements in VMEM and needs none. The price is TPU-only code, with the two-point
    flux rewritten in JAX.
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
  - An RK step takes 18.2 ms on a v5e, against about 33 ms for Trixi on an A100. One stage takes
    3.6 ms, which is about 650 GB/s, 80% of the v5e's peak. The whole run (317 steps) passes the GPU test.
  - Limits: the kernel covers only 3D Euler on brick meshes, and the whole time loop must be one
    compiled program, because each program boundary costs about 5 ms of layout copies.
  - It is worth continuing if TPU-only code is acceptable; the same fusion on an A100 comes first.
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
  - On the TPU a gather costs about 8 ns per index, so unstructured meshes take about 5× (3D) to
    14× (2D) longer per node than brick meshes. On an A100 the gather adds only 15%, and the array
    code is still 2.4× faster than Trixi (14.8 vs 36.0 ms).
  - I also tried tree blocks, with slices inside each tree and gathers between trees. They are
    correct, but faster than the plain gather (1.4×) only with $8^3$ elements per tree. AMR needs
    the gather path anyway.
  - This is worth continuing only if unstructured meshes on TPUs matter; for them a GPU is the
    simpler target.
]

= Upstream issues

== Issues reported to Reactant.jl

#let reactant(kind, n) = link("https://github.com/EnzymeAD/Reactant.jl/" + kind + "/" + str(n))[\##n]

I reported three issues, each with a minimal example (Reactant 0.2.297); pull requests fix two of
them:

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

#small[These issues have workarounds that do not change Trixi.]

== Other upstream problems

#[
  #set text(size: 15pt)
  - *KernelAbstractions*: Reactant cannot raise KA's bounds check for a 4D index range, which among
    Trixi's kernels only `FullSweepGlobal` uses. I worked around it by adding an unrolled `in`
    method, without changing Trixi. KA 0.10 probably fixes it with
    #link("https://github.com/JuliaGPU/KernelAbstractions.jl/pull/844")[\#844] (V. Churavy,
    10 October), which replaces the loop by a branch-free `reduce(&, ...)`. I have not tested this,
    because Reactant and Trixi still require KA 0.9.
  - *Not reported yet*: most problems with Trixi's own kernels in Reactant; the reported issues all
    come from the array code. With Reactant 0.2.291, kernel arguments that are types were rejected,
    the raised 2D interface kernel returned NaNs, and data-dependent loops, small local arrays,
    `sinpi` and barriers did not raise. Two problems are still present in 0.2.297 and have minimal
    examples: converting a `UInt8` to a float, and `reshape` inside a kernel.
]


= Summary and next steps

== Assessment: what is worth continuing

#[
  #show table: set text(size: 14pt)
  #table(
    columns: (1.25fr, 2.6fr, 1.05fr),
    [*Direction*], [*Result, compared with Trixi*], [*Continue?*],
    [array code (XLA)],
    [All P4est features tested (curved meshes, mortars, AMR, shock capturing) agree with Trixi to
      round-off; Trixi's GPU path has no mortars or AMR. On one v5e against Trixi on an A100, the
      weak form is on par, and flux differencing is 8.6× slower without Pallas.],
    [#ok as a package extension],
    [Pallas kernels],
    [For flux differencing, an RK step is 1.8× faster than Trixi on an A100. So far they cover only
      3D Euler on brick meshes, and they run only on TPUs.],
    [#ok yes, after the A100 comparison],
    [Trixi's own kernels through Reactant],
    [The raised flux-differencing kernel is correct, but on the TPU it is 23× slower than
      HalfSweep on an A100.],
    [#no no],
    [unstructured meshes on TPUs],
    [They take 5–14× longer per node than brick meshes; on an A100 the gather adds only 15%.],
    [#warn only if needed],
  )
]

#[
  #set text(size: 15pt)
  *My reading*: Trixi's P4est solver can run on TPUs with the features tested (curved meshes,
  mortars, AMR, shock capturing). Per chip, XLA alone matches
  Trixi's GPU kernels for the weak form; flux differencing passes them only with hand-written
  kernels, which are TPU-only.
]

== Why array code

#[
  #set text(size: 18pt)
  - XLA fuses the array operations into kernels.
  - On TPUs, Trixi's thread-style kernels need rewriting anyway: most did not raise (Reactant
    0.2.291), and the raised flux-differencing kernel takes 47.8 ms, against 2.19 ms for the Pallas
    kernel. So the array code is the frame, and Pallas kernels speed up the parts that take the
    most time.
  - On GPUs the array code is 2.8× faster than Trixi's one-thread-per-element kernel for the weak
    form, but 2.3× slower than HalfSweep for flux differencing.
]

== Next steps

#[
  #set text(size: 15pt)
  + *The same fusion on an A100*: put Trixi's HalfSweep, the surface terms, the sources and the RK
    update into one CUDA kernel, and compare it with the fused Pallas kernel (#repo-issue(12)). An
    A100 has 2.5× the bandwidth of a v5e, so if the kernel is bandwidth-bound, an RK step should take
    about 8 ms (v5e: 18.2 ms). A v5e costs €1.37/h, an A100 about \$1.2–1.8/h. This comparison
    decides whether TPUs pay off.
  + *Package extension*: turn the array code into a Reactant backend for Trixi's P4est solver.
  + *KA 0.10 with Reactant*: check that 4D kernels raise without the workaround (\#844).
  + *More Pallas kernels*: write them for other equations and unstructured meshes, but only if
    step 1 favours TPUs.
  + *Several TPU chips*: Reactant can split the arrays along the element dimension without code
    changes, so I expect near-linear scaling; this needs more TPU quota.
  + *AD*: not started. Reactant's claimed advantage, which I have not verified yet, is that Enzyme
    differentiates the whole traced tensor program (StableHLO) and XLA then optimizes the derivative
    like any other program. With KA, Enzyme differentiates each kernel separately in LLVM IR, with a
    tape per thread and atomic adds where threads read the same data.
]

== Open questions

#[
  #set text(size: 16pt)
  - Is TPU-only kernel code (Pallas) acceptable in a package extension?
  - *Code structure*: Trixi's KA kernels keep one structure (loops over elements) for CPUs,
    including MPI clusters, and GPUs. The array code is a second structure: whole-array operations
    that XLA fuses (the reasons are on the slide "Why array code"). Is that worth maintaining in
    Trixi?
  - *Time integration*: what did you mean by it on 13 October? Whole simulations already run with
    my own RK loop (Carpenter–Kennedy 2N54, as in Trixi's elixirs, with Trixi's CFL time step) and
    pass Trixi's GPU tests. Running Trixi's own `solve` (OrdinaryDiffEq) and its callbacks with
    Reactant is not done yet.
  - *AD*: which application could use derivatives? I could look for one in November.
]
