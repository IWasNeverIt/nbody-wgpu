use std::{fs::File, io::BufWriter};

/// Records simulation frames to an animated GIF.
///
/// Each frame is copied from the GPU surface texture to a staging buffer,
/// converted from BGRA/RGBA to RGB, then colour-quantised and appended.
pub struct GifRecorder {
    encoder:     gif::Encoder<BufWriter<File>>,
    pub staging: wgpu::Buffer,
    padded_row:  u32,
    pub width:   u32,
    pub height:  u32,
    total:       u32,
    pub remaining: u32,
    is_bgra:     bool,
}

impl GifRecorder {
    pub fn new(
        device:  &wgpu::Device,
        width:   u32,
        height:  u32,
        format:  wgpu::TextureFormat,
        frames:  u32,
        path:    &str,
    ) -> Self {
        // wgpu requires rows to be aligned to COPY_BYTES_PER_ROW_ALIGNMENT
        let align      = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded_row = (width * 4 + align - 1) / align * align;

        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("gif_staging"),
            size:               (padded_row * height) as u64,
            usage:              wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let file    = File::create(path).expect("cannot create nbody.gif");
        let mut enc = gif::Encoder::new(
            BufWriter::new(file), width as u16, height as u16, &[],
        ).unwrap();
        enc.set_repeat(gif::Repeat::Infinite).unwrap();

        let is_bgra = matches!(
            format,
            wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb
        );

        println!("Recording {} frames → {}", frames, path);
        Self { encoder: enc, staging, padded_row, width, height, total: frames, remaining: frames, is_bgra }
    }

    pub fn done(&self) -> bool { self.remaining == 0 }

    /// Encodes a copy command from the surface texture into the staging buffer.
    /// Must be called before `queue.submit`.
    pub fn copy_to_staging(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        texture: &wgpu::Texture,
    ) {
        encoder.copy_texture_to_buffer(
            texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &self.staging,
                layout: wgpu::TexelCopyBufferLayout {
                    offset:          0,
                    bytes_per_row:   Some(self.padded_row),
                    rows_per_image:  None,
                },
            },
            wgpu::Extent3d { width: self.width, height: self.height, depth_or_array_layers: 1 },
        );
    }

    /// Maps the staging buffer, colour-quantises the pixels, and appends a GIF frame.
    /// Must be called after `queue.submit` + `device.poll(wait_indefinitely)`.
    pub fn encode_frame(&mut self, device: &wgpu::Device) {
        if self.remaining == 0 { return; }

        let slice = self.staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();

        {
            let raw = slice.get_mapped_range();
            let mut rgb = Vec::with_capacity((self.width * self.height * 3) as usize);

            for y in 0..self.height as usize {
                let row = &raw[y * self.padded_row as usize..];
                for x in 0..self.width as usize {
                    let p = x * 4;
                    let (r, g, b) = if self.is_bgra {
                        (row[p + 2], row[p + 1], row[p])
                    } else {
                        (row[p], row[p + 1], row[p + 2])
                    };
                    rgb.push(r); rgb.push(g); rgb.push(b);
                }
            }

            // speed=10 is the fastest NeuQuant quantisation (lower quality but fine for previews)
            let mut frame = gif::Frame::from_rgb_speed(
                self.width as u16, self.height as u16, &rgb, 10,
            );
            frame.delay = 5; // 5 × 10 ms = 50 ms ≈ 20 fps
            self.encoder.write_frame(&frame).unwrap();
        }

        self.staging.unmap();
        self.remaining -= 1;

        let done  = self.total - self.remaining;
        if self.remaining == 0 {
            println!("GIF saved → nbody.gif  ({} frames)", self.total);
        } else if done % 10 == 0 {
            println!("  frame {}/{}", done, self.total);
        }
    }
}
