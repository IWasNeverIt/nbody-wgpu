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

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.n { return; }

    let pi = particles_in[i];
    var acc = vec2<f32>(0.0, 0.0);

    for (var j = 0u; j < params.n; j++) {
        if j == i { continue; }
        let pj  = particles_in[j];
        let d   = pj.pos - pi.pos;
        // softened gravitational acceleration: a = m_j * d / (|d|^2 + eps^2)^(3/2)
        let dist_sq    = dot(d, d) + params.softening * params.softening;
        let inv_dist3  = pow(dist_sq, -1.5);
        acc += pj.mass * d * inv_dist3;
    }

    // leapfrog-style symplectic Euler
    let vel = pi.vel + acc * params.dt;
    let pos = pi.pos + vel * params.dt;

    particles_out[i] = Particle(pos, vel, pi.mass, 0.0);
}
