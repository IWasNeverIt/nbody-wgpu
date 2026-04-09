mod capture;
mod compute;
pub mod cpu;
mod render;
mod sim;

use std::sync::Arc;
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalPosition,
    event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Icon, Window, WindowId},
};

/// Physics steps dispatched per rendered frame.
const STEPS_PER_FRAME: u32 = 4;

struct State {
    surface:       wgpu::Surface<'static>,
    device:        wgpu::Device,
    queue:         wgpu::Queue,
    config:        wgpu::SurfaceConfiguration,
    window:        Arc<Window>,
    sim:           sim::Simulation,
    renderer:      render::Renderer,
    gif:           Option<capture::GifRecorder>,
    should_exit:   bool,
    // camera
    pan:           [f32; 2],
    zoom:          f32,
    // mouse
    mouse_pressed: bool,
    last_cursor:   PhysicalPosition<f64>,
}

impl State {
    async fn new(window: Arc<Window>, n: u32, gif_frames: Option<u32>) -> Self {
        let instance = wgpu::Instance::default();
        let surface  = instance.create_surface(Arc::clone(&window)).unwrap();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference:   wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                ..Default::default()
            })
            .await
            .expect("no GPU adapter found");

        println!("Adapter: {}", adapter.get_info().name);

        let (device, queue) = adapter
            .request_device(&Default::default())
            .await
            .unwrap();

        compute::run_double_test(&device, &queue);

        let sim = sim::Simulation::new(&device, &queue, n);
        println!("Simulation: {} particles, {} steps/frame", sim.n, STEPS_PER_FRAME);

        let size   = window.inner_size();
        let caps   = surface.get_capabilities(&adapter);
        let format = caps.formats[0];

        // COPY_SRC is needed when recording a GIF (copy surface → staging buffer)
        let mut usage = wgpu::TextureUsages::RENDER_ATTACHMENT;
        if gif_frames.is_some() { usage |= wgpu::TextureUsages::COPY_SRC; }

        let config = wgpu::SurfaceConfiguration {
            usage,
            format,
            width:   size.width,
            height:  size.height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode:   caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        let bufs     = sim.buffers();
        let renderer = render::Renderer::new(&device, format, sim.n, bufs[0], bufs[1]);

        let (pan, zoom) = ([0.0_f32, 0.0], 1.05_f32);
        renderer.update_camera(&queue, pan, zoom);

        let gif = gif_frames.map(|frames| {
            capture::GifRecorder::new(
                &device, size.width, size.height, format, frames, "nbody.gif",
            )
        });

        Self {
            surface, device, queue, config, window,
            sim, renderer, gif,
            should_exit: false,
            pan, zoom,
            mouse_pressed: false,
            last_cursor:   PhysicalPosition::default(),
        }
    }

    fn render(&mut self) {
        let output = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t) | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
            _ => return,
        };
        let view = output.texture.create_view(&Default::default());
        let mut enc = self.device.create_command_encoder(&Default::default());

        for _ in 0..STEPS_PER_FRAME {
            self.sim.step(&mut enc);
        }
        self.renderer.draw(&mut enc, &view, self.sim.cur());

        if let Some(gif) = &self.gif {
            gif.copy_to_staging(&mut enc, &output.texture);
        }

        self.queue.submit([enc.finish()]);

        if let Some(gif) = &mut self.gif {
            gif.encode_frame(&self.device);
            if gif.done() { self.should_exit = true; }
        }

        output.present();
        self.window.request_redraw();
    }

    fn on_scroll(&mut self, delta: MouseScrollDelta) {
        let lines = match delta {
            MouseScrollDelta::LineDelta(_, y)  => y,
            MouseScrollDelta::PixelDelta(p)    => p.y as f32 * 0.05,
        };
        self.zoom = (self.zoom * 1.15_f32.powf(lines)).clamp(0.05, 100.0);
        self.renderer.update_camera(&self.queue, self.pan, self.zoom);
    }

    fn on_cursor_moved(&mut self, pos: PhysicalPosition<f64>) {
        if self.mouse_pressed {
            let size = self.window.inner_size();
            let dx =  (pos.x - self.last_cursor.x) as f32 / (size.width  as f32 * 0.5);
            let dy = -(pos.y - self.last_cursor.y) as f32 / (size.height as f32 * 0.5);
            self.pan[0] += dx / self.zoom;
            self.pan[1] += dy / self.zoom;
            self.renderer.update_camera(&self.queue, self.pan, self.zoom);
        }
        self.last_cursor = pos;
    }
}

struct App {
    state:      Option<State>,
    n:          u32,
    gif_frames: Option<u32>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = Arc::new(
            event_loop.create_window(Default::default()).unwrap()
        );
        window.set_window_icon(Some(make_window_icon()));
        self.state = Some(pollster::block_on(State::new(window, self.n, self.gif_frames)));
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(s) = &mut self.state else { return };
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::RedrawRequested => {
                s.render();
                if s.should_exit { event_loop.exit(); }
            }

            WindowEvent::MouseInput { button: MouseButton::Left, state, .. } =>
                s.mouse_pressed = state == ElementState::Pressed,

            WindowEvent::CursorMoved { position, .. } => s.on_cursor_moved(position),
            WindowEvent::MouseWheel  { delta, .. }    => s.on_scroll(delta),
            _ => {}
        }
    }
}

fn main() {
    env_logger::init();
    let (n, gif_frames) = parse_args();
    EventLoop::new().unwrap()
        .run_app(&mut App { state: None, n, gif_frames })
        .unwrap();
}

fn parse_args() -> (u32, Option<u32>) {
    let args: Vec<String> = std::env::args().collect();
    let mut n   = sim::N_DEFAULT;
    let mut gif = None;
    let mut i   = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--n" if i + 1 < args.len() => {
                n   = args[i + 1].parse().unwrap_or(n).max(4);
                i  += 2;
            }
            "--gif" => {
                let frames = if i + 1 < args.len() {
                    args[i + 1].parse().unwrap_or(90)
                } else { 90 };
                gif = Some(frames);
                i  += if i + 1 < args.len() && args[i+1].parse::<u32>().is_ok() { 2 } else { 1 };
            }
            _ => i += 1,
        }
    }
    (n, gif)
}

/// Generates a 64×64 window icon: two glowing galactic cores + star field.
fn make_window_icon() -> Icon {
    const S: u32 = 64;
    let mut rgba = vec![0u8; (S * S * 4) as usize];
    let cores: &[(f32, f32)] = &[(43.0, 16.0), (20.0, 47.0)];

    for y in 0..S {
        for x in 0..S {
            let idx = ((y * S + x) * 4) as usize;
            let mut v = 0.0f32;
            for &(cx, cy) in cores {
                let dx = x as f32 - cx;
                let dy = y as f32 - cy;
                v += (-(dx * dx + dy * dy) / 30.0).exp();
            }
            let h    = x.wrapping_mul(2654435761).wrapping_add(y.wrapping_mul(2246822519));
            let star = if h & 0xFFFF < 180 { 0.18 } else { 0.0 };
            let b    = ((v + star).min(1.0) * 255.0) as u8;
            rgba[idx] = b; rgba[idx+1] = b; rgba[idx+2] = b; rgba[idx+3] = 255;
        }
    }
    Icon::from_rgba(rgba, S, S).unwrap()
}
