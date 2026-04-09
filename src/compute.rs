const SHADER: &str = r#"
@group(0) @binding(0) var<storage, read_write> data: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i < arrayLength(&data) {
        data[i] = data[i] * 2.0;
    }
}
"#;

pub fn run_double_test(device: &wgpu::Device, queue: &wgpu::Queue) {
    const N: usize = 8;
    let input: Vec<f32> = (1..=N as u32).map(|x| x as f32).collect();

    // Storage buffer (GPU read/write)
    let storage_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("storage"),
        size: (N * std::mem::size_of::<f32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&storage_buf, 0, bytemuck::cast_slice(&input));

    // Staging buffer (CPU read)
    let staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging"),
        size: storage_buf.size(),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Pipeline
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("double"),
        source: wgpu::ShaderSource::Wgsl(SHADER.into()),
    });
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("double"),
        layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&bgl)],
            ..Default::default()
        })),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: storage_buf.as_entire_binding(),
        }],
    });

    // Dispatch
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(((N as u32) + 63) / 64, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&storage_buf, 0, &staging_buf, 0, storage_buf.size());
    queue.submit([encoder.finish()]);

    // Read back
    let slice = staging_buf.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();

    let result: Vec<f32> = bytemuck::cast_slice(&slice.get_mapped_range()).to_vec();
    print!("compute test — input:  {:?}\n", input);
    print!("compute test — output: {:?}\n", result);
    assert!(
        result.iter().zip(input.iter()).all(|(r, i)| (*r - i * 2.0).abs() < 1e-6),
        "GPU double test failed!"
    );
    println!("compute test — PASSED");
}
