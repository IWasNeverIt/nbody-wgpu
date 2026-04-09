// Instanced point-sprite rendering.
// 6 vertices per quad × N instances; vertex shader reads particles from storage buffer.

struct Particle {
    pos:  vec2<f32>,
    vel:  vec2<f32>,
    mass: f32,
    _pad: f32,
}

struct Camera {
    pan:  vec2<f32>,
    zoom: f32,
    _pad: f32,
}

@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<uniform>       camera:    Camera;

// CCW quad — two triangles sharing the (-1,-1)→(1,1) diagonal
const QUAD: array<vec2<f32>, 6> = array(
    vec2(-1.0, -1.0), vec2( 1.0, -1.0), vec2( 1.0,  1.0),
    vec2(-1.0, -1.0), vec2( 1.0,  1.0), vec2(-1.0,  1.0),
);
const POINT_RADIUS: f32 = 0.003; // in clip-space units

struct VertOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0)       speed:    f32,
    @location(1)       uv:       vec2<f32>, // -1..1, for circle clip
}

@vertex
fn vs_main(
    @builtin(vertex_index)   vi: u32,
    @builtin(instance_index) ii: u32,
) -> VertOut {
    let p      = particles[ii];
    let corner = QUAD[vi];
    let center = (p.pos + camera.pan) * camera.zoom;
    let clip   = center + corner * POINT_RADIUS;

    return VertOut(
        vec4<f32>(clip, 0.0, 1.0),
        length(p.vel),
        corner,
    );
}

@fragment
fn fs_main(in: VertOut) -> @location(0) vec4<f32> {
    // discard corners to get a circle
    if dot(in.uv, in.uv) > 1.0 { discard; }

    // blue (slow) → white-orange (fast); alpha kept low so additive blending glows
    let t     = clamp(in.speed / 120.0, 0.0, 1.0);
    let color = mix(vec3(0.20, 0.40, 1.0), vec3(1.0, 0.75, 0.3), t);
    return vec4<f32>(color, 0.55);
}
