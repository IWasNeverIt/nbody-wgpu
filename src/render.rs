#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Camera {
    pan:  [f32; 2],
    zoom: f32,
    _pad: f32,
}

pub struct Renderer {
    pipeline:    wgpu::RenderPipeline,
    bind_groups: [wgpu::BindGroup; 2], // one per ping-pong buffer
    camera_buf:  wgpu::Buffer,
    n:           u32,
}

impl Renderer {
    pub fn new(
        device:  &wgpu::Device,
        format:  wgpu::TextureFormat,
        n:       u32,
        buf_a:   &wgpu::Buffer,
        buf_b:   &wgpu::Buffer,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("render"),
            source: wgpu::ShaderSource::Wgsl(include_str!("render.wgsl").into()),
        });

        let camera = Camera { pan: [0.0, 0.0], zoom: 1.05, _pad: 0.0 };
        let camera_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("camera"),
            size:               std::mem::size_of::<Camera>() as u64,
            usage:              wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label:   None,
            entries: &[
                // particles storage buffer
                wgpu::BindGroupLayoutEntry {
                    binding:    0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty:                 wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size:   None,
                    },
                    count: None,
                },
                // camera uniform
                wgpu::BindGroupLayoutEntry {
                    binding:    1,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty:                 wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size:   None,
                    },
                    count: None,
                },
            ],
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label:               None,
            bind_group_layouts:  &[Some(&bgl)],
            ..Default::default()
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label:  Some("render"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module:             &shader,
                entry_point:        Some("vs_main"),
                buffers:            &[], // no vertex buffer; reads storage buffer directly
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module:      &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    // Additive blending: overlapping particles accumulate brightness,
                    // producing a natural "glow" wherever stars cluster.
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::SrcAlpha,
                            dst_factor: wgpu::BlendFactor::One,
                            operation:  wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::One,
                            operation:  wgpu::BlendOperation::Add,
                        },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample:   wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache:         None,
        });

        let make_bg = |buf: &wgpu::Buffer| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label:   None,
                layout:  &bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: camera_buf.as_entire_binding() },
                ],
            })
        };

        // Write initial camera to GPU
        let camera_bytes = bytemuck::bytes_of(&camera);
        // We'll use queue.write_buffer in draw() — store the initial value for now.
        // Actually write via a small trick: we need queue here. Let's accept queue.
        // Instead, we'll just write zeros and let the caller call update_camera().
        // Actually simpler: accept queue as a param.
        let _ = camera_bytes; // written via update_camera after construction

        Self {
            pipeline,
            bind_groups: [make_bg(buf_a), make_bg(buf_b)],
            camera_buf,
            n,
        }
    }

    /// Must be called once after `new` to upload the initial camera, and any time it changes.
    pub fn update_camera(&self, queue: &wgpu::Queue, pan: [f32; 2], zoom: f32) {
        let cam = Camera { pan, zoom, _pad: 0.0 };
        queue.write_buffer(&self.camera_buf, 0, bytemuck::bytes_of(&cam));
    }

    /// Encode the render pass into `encoder`. `cur` must match `Simulation::cur()`.
    pub fn draw(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        view:    &wgpu::TextureView,
        cur:     usize,
    ) {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("render"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                depth_slice:    None,
                ops: wgpu::Operations {
                    load:  wgpu::LoadOp::Clear(wgpu::Color { r: 0.0, g: 0.0, b: 0.0, a: 1.0 }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            ..Default::default()
        });

        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_groups[cur], &[]);
        // 6 verts per quad, N instances
        pass.draw(0..6, 0..self.n);
    }
}
