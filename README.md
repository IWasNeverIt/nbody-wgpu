# nbody-wgpu

GPU-accelerated N-body gravitational simulation written in Rust using [wgpu](https://wgpu.rs/).

All-pairs O(N²) gravity runs entirely on the GPU via a tiled WGSL compute shader.
Two counter-rotating disk galaxies collide and form tidal tails in real time.
Particles glow blue→orange by velocity magnitude via additive blending.

## Build & run

```
cargo run --release
cargo run --release -- --n 4096
```

Requires a GPU with Vulkan, Metal, or DirectX 12 support.

## Controls

| Action | Effect |
|---|---|
| Left-drag | Pan |
| Scroll wheel | Zoom in / out |
| `--n <count>` | Number of particles (default: 1024) |

## Architecture

```
main.rs          window + event loop (winit), mouse/scroll camera controls
sim.rs           Simulation — ping-pong compute buffers, nbody.wgsl dispatch
render.rs        Renderer   — instanced point-sprite pipeline, render.wgsl
compute.rs       one-shot GPU round-trip test (runs at startup)
cpu.rs           CPU O(N²/2) reference step using Newton's 3rd law
nbody.wgsl       tiled all-pairs gravity compute shader (@workgroup_size 64)
render.wgsl      vertex (storage buffer → clip space) + fragment (velocity → color)
benches/nbody.rs Criterion benchmarks — CPU step at N = 256 / 1k / 4k / 16k
```

## How it works

Each frame, 4 physics steps are dispatched before rendering:

1. **Compute pass (tiled)** — the physics shader divides the particle array into
   64-particle tiles. Each workgroup cooperatively loads one tile into
   `var<workgroup>` shared memory, then all 64 threads accumulate forces from
   it before moving to the next tile. This reduces global memory reads by 64×
   compared to a naive kernel, turning a bandwidth-bound problem into a
   compute-bound one.  
   Integration: symplectic Euler (`vel += a·dt`, `pos += vel·dt`).  
   Two storage buffers alternate each frame (ping-pong) so reads and writes never race.

2. **Render pass** — particles are drawn as instanced quads (6 verts × N instances).  
   The vertex shader reads directly from the compute output buffer via a `BindGroup`
   — no CPU readback.  
   Additive blending (`dst + src·α`) makes overlapping particles accumulate
   brightness, giving a natural glow at galactic cores and along tidal streams.  
   Fragment color maps velocity magnitude to a blue→orange gradient.

## Performance (preliminary)

| N | GPU (RTX 3060 Ti) | CPU (reference) |
|---|---|---|
| 256 | — | — |
| 1 024 | — | — |
| 4 096 | — | — |
| 16 384 | — | — |

Run `cargo bench` to populate this table. The CPU crossover (where GPU overhead
is justified) is expected around N ≈ 500–1000.

## Roadmap

- [x] Days 1–2 · wgpu boilerplate — window, adapter, device, render loop
- [x] Day 3 · first compute shader — GPU round-trip proof of concept
- [x] Days 4–5 · N-body WGSL shader — all-pairs gravity + symplectic Euler
- [x] Days 6–7 · buffer ping-pong — race-free double-buffering
- [x] Days 8–9 · render pipeline — instanced point sprites, first particles visible
- [x] Days 10–11 · visual polish — velocity color, additive glow, pan + zoom, galaxy collision
- [x] Days 12–13 · CPU benchmark — Criterion harness, `cpu.rs` reference implementation
- [ ] Day 14 · fill perf table, add GIF demo
