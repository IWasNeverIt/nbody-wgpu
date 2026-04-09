use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::f32::consts::TAU;

// ── Particle data (mirrors sim::Particle layout, no wgpu dep needed) ─────────

struct Particles {
    pos:  Vec<[f32; 2]>,
    vel:  Vec<[f32; 2]>,
    mass: Vec<f32>,
}

impl Particles {
    fn galaxy_collision(n: usize) -> Self {
        let half = n / 2;
        let (mut pos, mut vel, mut mass) = (vec![], vec![], vec![]);
        for (count, cx, cy, dvx, dvy, spin) in [
            (half,      -0.50_f32,  0.18_f32,  0.22_f32, -0.04_f32,  1.0_f32),
            (n - half,   0.50,     -0.18,      -0.22,      0.04,     -1.0),
        ] {
            let m_c = 500.0_f32;
            pos.push([cx, cy]); vel.push([dvx, dvy]); mass.push(m_c);
            for i in 1..count {
                let t = i as f32 / (count - 1) as f32;
                let a = t * TAU * 6.0;
                let r = 0.03 + t * 0.32;
                let s = (m_c / r).sqrt() * spin;
                pos.push([cx + r * a.cos(), cy + r * a.sin()]);
                vel.push([dvx - s * a.sin(), dvy + s * a.cos()]);
                mass.push(1.0);
            }
        }
        Self { pos, vel, mass }
    }
}

// ── CPU O(N²/2) step ─────────────────────────────────────────────────────────

fn cpu_step(p: &mut Particles, dt: f32, soft: f32) {
    let n = p.pos.len();
    let mut ax = vec![0.0_f32; n];
    let mut ay = vec![0.0_f32; n];

    for i in 0..n {
        for j in (i + 1)..n {
            let dx   = p.pos[j][0] - p.pos[i][0];
            let dy   = p.pos[j][1] - p.pos[i][1];
            let inv3 = (dx * dx + dy * dy + soft * soft).powf(-1.5);
            ax[i] += p.mass[j] * dx * inv3;
            ay[i] += p.mass[j] * dy * inv3;
            ax[j] -= p.mass[i] * dx * inv3;
            ay[j] -= p.mass[i] * dy * inv3;
        }
    }
    for i in 0..n {
        p.vel[i][0] += ax[i] * dt;
        p.vel[i][1] += ay[i] * dt;
        p.pos[i][0] += p.vel[i][0] * dt;
        p.pos[i][1] += p.vel[i][1] * dt;
    }
}

// ── Benchmarks ───────────────────────────────────────────────────────────────

fn bench_cpu(c: &mut Criterion) {
    let mut group = c.benchmark_group("cpu_step");
    group.sample_size(10); // O(N²) is slow at large N

    for &n in &[256_usize, 1_024, 4_096, 16_384] {
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            let mut p = Particles::galaxy_collision(n);
            b.iter(|| cpu_step(&mut p, 0.0005, 0.05));
        });
    }
    group.finish();
}

criterion_group!(benches, bench_cpu);
criterion_main!(benches);
