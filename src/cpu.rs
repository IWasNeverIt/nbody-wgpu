/// CPU O(N²) n-body reference implementation.
///
/// Uses Newton's 3rd law to halve the work: each (i, j) pair is computed once
/// and the equal-and-opposite acceleration applied to both particles.
pub fn step(pos: &mut [[f32; 2]], vel: &mut [[f32; 2]], mass: &[f32], dt: f32, softening: f32) {
    let n = pos.len();
    let mut ax = vec![0.0_f32; n];
    let mut ay = vec![0.0_f32; n];

    for i in 0..n {
        for j in (i + 1)..n {
            let dx = pos[j][0] - pos[i][0];
            let dy = pos[j][1] - pos[i][1];
            let inv3 = (dx * dx + dy * dy + softening * softening).powf(-1.5);

            ax[i] += mass[j] * dx * inv3;
            ay[i] += mass[j] * dy * inv3;
            ax[j] -= mass[i] * dx * inv3;
            ay[j] -= mass[i] * dy * inv3;
        }
    }

    for i in 0..n {
        vel[i][0] += ax[i] * dt;
        vel[i][1] += ay[i] * dt;
        pos[i][0] += vel[i][0] * dt;
        pos[i][1] += vel[i][1] * dt;
    }
}
