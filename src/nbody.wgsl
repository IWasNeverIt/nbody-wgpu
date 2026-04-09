// Tiled all-pairs N-body gravity.
//
// Each workgroup loads a 64-particle tile into workgroup-shared memory, then
// every thread accumulates forces from that tile before moving to the next.
// This reduces global memory reads by a factor of TILE (64) compared to the
// naive kernel — turning a bandwidth-bound problem into a compute-bound one.

struct Particle {
    pos:  vec2<f32>,
    vel:  vec2<f32>,
    mass: f32,
    _pad: f32,
}

struct Params {
    n:         u32,
    dt:        f32,
    softening: f32,
    _pad:      f32,
}

@group(0) @binding(0) var<storage, read>       particles_in:  array<Particle>;
@group(0) @binding(1) var<storage, read_write> particles_out: array<Particle>;
@group(0) @binding(2) var<uniform>             params:        Params;

const TILE: u32 = 64u;

// Workgroup-shared tile — all threads in a workgroup read from this
// instead of hitting global memory for every (i, j) pair.
var<workgroup> tile: array<Particle, 64>;

@compute @workgroup_size(64)
fn main(
    @builtin(global_invocation_id)   gid: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    let i      = gid.x;
    let active = i < params.n;

    var pi:  Particle;
    var acc = vec2<f32>(0.0);
    if active { pi = particles_in[i]; }

    let num_tiles = (params.n + TILE - 1u) / TILE;

    for (var t = 0u; t < num_tiles; t++) {
        // --- Cooperative load: each thread fetches one particle into shared mem ---
        let j = t * TILE + lid;
        if j < params.n {
            tile[lid] = particles_in[j];
        } else {
            // Pad with a zero-mass ghost so arithmetic stays uniform (no branches
            // in the inner loop, no divergence, no out-of-bounds force).
            tile[lid] = Particle(vec2(0.0), vec2(0.0), 0.0, 0.0);
        }
        workgroupBarrier(); // ensure all threads finished writing before any reads

        // --- Accumulate forces from this tile ---
        // Self-force: d = (0,0), so force = mass * (0,0) * inv_dist³ = 0. Safe.
        // Ghost-force: mass = 0, so force = 0. Safe.
        // No branch needed in the inner loop — both cases produce 0 naturally.
        if active {
            for (var k = 0u; k < TILE; k++) {
                let pj       = tile[k];
                let d        = pj.pos - pi.pos;
                let dist_sq  = dot(d, d) + params.softening * params.softening;
                acc += pj.mass * d * pow(dist_sq, -1.5);
            }
        }
        workgroupBarrier(); // done reading; safe for next tile load
    }

    if active {
        let vel = pi.vel + acc * params.dt;
        let pos = pi.pos + vel * params.dt;
        particles_out[i] = Particle(pos, vel, pi.mass, 0.0);
    }
}
