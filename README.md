# nbody-wgpu

GPU-accelerated N-body gravitational simulation written in Rust using [wgpu](https://wgpu.rs/).

All-pairs O(N²) gravity runs entirely on the GPU via a WGSL compute shader.
Particles are rendered as point sprites colored by velocity magnitude (warm = fast, cool = slow).

## Build & run

```
cargo run --release
```

Requires a GPU with Vulkan, Metal, or DirectX 12 support.

## Controls

| Key / Action | Effect |
|---|---|
| `--n <count>` | Number of particles (default: 1024) |

Pan and zoom coming in Days 10–11.

## Architecture

```
main.rs          window + event loop (winit)
sim.rs           Simulation — ping-pong compute buffers, nbody.wgsl dispatch
render.rs        Renderer   — instanced point-sprite pipeline, render.wgsl
compute.rs       one-shot GPU round-trip test (runs at startup)
nbody.wgsl       all-pairs gravity compute shader  (@workgroup_size 64)
render.wgsl      vertex (storage buffer → clip space) + fragment (velocity → color)
benches/nbody.rs Criterion bench harness (CPU vs GPU crossover, Days 12–13)
```

## How it works

Each frame:

1. **Compute pass** — the physics shader runs one time-step.  
   Thread `i` accumulates softened gravitational acceleration from every other particle, then integrates with symplectic Euler (`vel += a·dt`, `pos += vel·dt`).  
   Two storage buffers alternate each frame (ping-pong) so reads and writes never race.

2. **Render pass** — particles are drawn as instanced quads (6 verts × N instances).  
   The vertex shader reads directly from the compute output buffer via a `BindGroup`; no CPU readback needed.  
   Fragment color maps velocity magnitude to a blue–orange gradient.

## Performance (preliminary)

| N | GPU (RTX ?) | CPU (reference) |
|---|---|---|
| 256 | — | — |
| 1 024 | — | — |
| 4 096 | — | — |
| 16 384 | — | — |

_Benchmark results coming in Days 12–13 (`cargo bench`)._

## Roadmap

- [x] Days 1–2 · wgpu boilerplate — window, adapter, device, render loop
- [x] Day 3 · first compute shader — GPU round-trip proof of concept
- [x] Days 4–5 · N-body WGSL shader — all-pairs gravity + symplectic Euler
- [x] Days 6–7 · buffer ping-pong — race-free double-buffering
- [ ] Days 8–9 · render pipeline — point sprites, first time you see particles
- [ ] Days 10–11 · visual polish — velocity color, 2D pan + zoom
- [ ] Days 12–13 · CPU benchmark — Criterion, find GPU crossover N
- [ ] Day 14 · README perf graph + GIF, CLI flags, GitHub push
