mod compute;
mod render;
mod sim;

use std::sync::Arc;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowId},
};

struct State {
    surface:  wgpu::Surface<'static>,
    device:   wgpu::Device,
    queue:    wgpu::Queue,
    config:   wgpu::SurfaceConfiguration,
    window:   Arc<Window>,
    sim:      sim::Simulation,
    renderer: render::Renderer,
}

impl State {
    async fn new(window: Arc<Window>) -> Self {
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

        println!("Adapter: {:?}", adapter.get_info().name);

        let (device, queue) = adapter
            .request_device(&Default::default())
            .await
            .unwrap();

        compute::run_double_test(&device, &queue);

        let sim = sim::Simulation::new(&device, &queue, sim::N_DEFAULT);
        println!("Simulation: {} particles", sim.n);

        let size = window.inner_size();
        let caps = surface.get_capabilities(&adapter);
        let format = caps.formats[0];
        let config = wgpu::SurfaceConfiguration {
            usage:   wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width:   size.width,
            height:  size.height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode:   caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        let bufs = sim.buffers();
        let renderer = render::Renderer::new(&device, format, sim.n, bufs[0], bufs[1]);
        renderer.update_camera(&queue, [0.0, 0.0], 1.05);

        Self { surface, device, queue, config, window, sim, renderer }
    }

    fn render(&mut self) {
        let output = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t) | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
            _ => return,
        };
        let view    = output.texture.create_view(&Default::default());
        let mut encoder = self.device.create_command_encoder(&Default::default());

        self.sim.step(&mut encoder);
        self.renderer.draw(&mut encoder, &view, self.sim.cur());

        self.queue.submit([encoder.finish()]);
        output.present();
        self.window.request_redraw();
    }
}

#[derive(Default)]
struct App {
    state: Option<State>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = Arc::new(
            event_loop.create_window(Default::default()).unwrap()
        );
        self.state = Some(pollster::block_on(State::new(window)));
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::RedrawRequested => {
                if let Some(s) = &mut self.state { s.render(); }
            }
            _ => {}
        }
    }
}

fn main() {
    env_logger::init();
    EventLoop::new().unwrap().run_app(&mut App::default()).unwrap();
}
