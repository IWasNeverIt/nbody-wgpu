use std::f32::consts::TAU;

pub const N_DEFAULT: u32 = 1024;

/// Matches the WGSL `Particle` struct layout exactly (24 bytes).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Particle {
    pub pos:  [f32; 2],
    pub vel:  [f32; 2],
    pub mass: f32,
    pub _pad: f32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    n:         u32,
    dt:        f32,
    softening: f32,
    _pad:      f32,
}

pub struct Simulation {
    pub n:       u32,
    bufs:        [wgpu::Buffer; 2],
    bind_groups: [wgpu::BindGroup; 2],
    pipeline:    wgpu::ComputePipeline,
    /// Index of the buffer that was written last (holds current state).
    cur: usize,
}

impl Simulation {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, n: u32) -> Self {
        let particles = init_particles(n);
        let buf_size  = (n as usize * std::mem::size_of::<Particle>()) as u64;

        let buf_usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;

        let buf_a = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("particles_a"), size: buf_size, usage: buf_usage,
            mapped_at_creation: false,
        });
        let buf_b = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("particles_b"), size: buf_size, usage: buf_usage,
            mapped_at_creation: false,
        });
        queue.write_buffer(&buf_a, 0, bytemuck::cast_slice(&particles));

        let params     = Params { n, dt: 0.0005, softening: 0.05, _pad: 0.0 };
        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("params"),
            size:  std::mem::size_of::<Params>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&params_buf, 0, bytemuck::bytes_of(&params));

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("nbody"),
            source: wgpu::ShaderSource::Wgsl(include_str!("nbody.wgsl").into()),
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label:   None,
            entries: &[
                storage_bgl_entry(0, false), // particles_in  (read-only)
                storage_bgl_entry(1, true),  // particles_out (read-write)
                uniform_bgl_entry(2),        // params
            ],
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&bgl)],
            ..Default::default()
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label:               Some("nbody"),
            layout:              Some(&layout),
            module:              &shader,
            entry_point:         Some("main"),
            compilation_options: Default::default(),
            cache:               None,
        });

        // bind_groups[0]: buf_a → in,  buf_b → out
        // bind_groups[1]: buf_b → in,  buf_a → out
        let bind_groups = [
            make_bind_group(device, &bgl, &buf_a, &buf_b, &params_buf),
            make_bind_group(device, &bgl, &buf_b, &buf_a, &params_buf),
        ];

        Self { n, bufs: [buf_a, buf_b], bind_groups, pipeline, cur: 0 }
    }

    /// Encode one physics step into `encoder`. Call before the render pass.
    pub fn step(&mut self, encoder: &mut wgpu::CommandEncoder) {
        let next = 1 - self.cur;
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_groups[self.cur], &[]);
        pass.dispatch_workgroups((self.n + 63) / 64, 1, 1);
        drop(pass);
        self.cur = next;
    }

    /// Index of the buffer holding the most recently written particle state.
    pub fn cur(&self) -> usize { self.cur }

    /// Both ping-pong buffers, in order. Use `cur()` to know which is current.
    pub fn buffers(&self) -> [&wgpu::Buffer; 2] {
        [&self.bufs[0], &self.bufs[1]]
    }
}

// ── helpers ──────────────────────────────────────────────────────────────────

fn storage_bgl_entry(binding: u32, read_write: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: !read_write },
            has_dynamic_offset: false,
            min_binding_size:   None,
        },
        count: None,
    }
}

fn uniform_bgl_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty:                wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size:   None,
        },
        count: None,
    }
}

fn make_bind_group(
    device:     &wgpu::Device,
    layout:     &wgpu::BindGroupLayout,
    buf_in:     &wgpu::Buffer,
    buf_out:    &wgpu::Buffer,
    params_buf: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label:   None,
        layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: buf_in.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: buf_out.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: params_buf.as_entire_binding() },
        ],
    })
}

/// Central heavy mass + lighter particles on circular orbits.
fn init_particles(n: u32) -> Vec<Particle> {
    let mut out = Vec::with_capacity(n as usize);

    // Central body
    out.push(Particle { pos: [0.0, 0.0], vel: [0.0, 0.0], mass: 1000.0, _pad: 0.0 });

    for i in 1..n {
        let t     = i as f32 / (n - 1) as f32;
        let angle = t * TAU * 7.0;          // spread across several windings
        let r     = 0.15 + t * 0.75;        // 0.15 .. 0.90
        let speed = (1000.0_f32 / r).sqrt(); // circular-orbit speed (G=1, M=1000)

        out.push(Particle {
            pos:  [r * angle.cos(), r * angle.sin()],
            vel:  [-speed * angle.sin(), speed * angle.cos()],
            mass: 1.0,
            _pad: 0.0,
        });
    }
    out
}
