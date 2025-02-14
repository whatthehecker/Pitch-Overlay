mod crepe;

use cpal::traits::{DeviceTrait, HostTrait};
use cpal::{BufferSize, Device, SampleRate, Stream, StreamConfig, StreamInstant};
use eframe::egui::{Color32, Context, Label, RichText};
use eframe::{egui, Frame};
use egui_plot::{HLine, Line, Plot, PlotBounds, PlotPoints};
use std::sync::{Arc, RwLock};
use std::time::Duration;

const CONFIG: StreamConfig = StreamConfig {
    channels: 1,
    sample_rate: SampleRate(16_000),
    // 1024 is 64 milliseconds, the smallest unit that CREPE can handle.
    buffer_size: BufferSize::Fixed(1024),
};

fn main() -> eframe::Result {
    /*
    ort::init()
        .commit()
        .expect("Failed to init ort.");
    let session = Session::builder().unwrap()
        .commit_from_file("assets/crepe-full.onnx").unwrap();
    let crepe_model = CrepeModel::new(session);
    
     */

    let host = cpal::default_host();
    let all_devices = host.input_devices().expect("Failed to get input devices").map(|device| device.clone()).collect::<Vec<Device>>();

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([320.0, 240.0]),
        ..Default::default()
    };
    eframe::run_native(
        "My egui App",
        options,
        Box::new(|cc| {
            egui_extras::install_image_loaders(&cc.egui_ctx);

            Ok(Box::<MyApp>::new(MyApp::new(
                all_devices,
                //crepe_model,
            )))
        }),
    )
}

struct AudioState {
    first_audio_instant: Option<StreamInstant>,
    pitch_points: Vec<[f64; 2]>,
}

impl Default for AudioState {
    fn default() -> Self {
        AudioState {
            first_audio_instant: None,
            pitch_points: vec![],
        }
    }
}

struct MyApp {
    name: String,
    age: u32,
    current_stream: Option<Stream>,
    current_device_index: Option<usize>,
    available_input_devices: Vec<Device>,

    audio_state: Arc<RwLock<AudioState>>,

    //crepe_model: Arc<CrepeModel>,

    // TODO: turn these into [u32; 2]
    min_display_frequency: u32,
    max_display_frequency: u32,
    min_target_frequency: u32,
    max_target_frequency: u32,

    settings_open: bool,
}

impl MyApp {
    fn new(input_devices: Vec<Device>/*, crepe_model: CrepeModel */) -> Self {
        Self {
            name: "Arthur".to_owned(),
            age: 42,
            current_stream: None,
            current_device_index: None,
            available_input_devices: input_devices,

            audio_state: Arc::new(RwLock::new(AudioState::default())),

            //crepe_model: Arc::new(crepe_model),

            min_display_frequency: 50,
            max_display_frequency: 500,
            min_target_frequency: 185,
            max_target_frequency: 300,

            settings_open: false,
        }
    }

    fn current_device(&self) -> Option<Device> {
        if let Some(i) = self.current_device_index {
            return Some(self.available_input_devices[i].clone());
        }

        None
    }
}

impl eframe::App for MyApp {
    fn update(&mut self, ctx: &Context, _frame: &mut Frame) {
        if self.settings_open {
            let current_device_name = self.current_device().map(|device| device.name().unwrap_or("Unnamed device".to_owned())).unwrap_or("Audio disconnected".to_owned());

            egui::Window::new("Settings")
                .collapsible(false)
                .open(&mut self.settings_open)
                .show(ctx, |ui| {
                    egui::ComboBox::from_label("Audio Input device")
                        .selected_text(format!("{}", current_device_name))
                        .show_ui(ui, |ui| {
                            if ui.selectable_value(&mut self.current_device_index, None, "Disconnect audio").clicked() {
                                println!("Disconnect clicked!");
                                self.current_stream = None;
                            }
                            for (i, device) in self.available_input_devices.iter().enumerate() {
                                let name = device.name().unwrap_or("Unknown device".to_owned());
                                if ui.selectable_value(&mut self.current_device_index, Some(i), name).clicked() {
                                    println!("Connect to new device clicked!");

                                    let cloned_arc = Arc::clone(&self.audio_state);
                                    let cloned_ctx = ctx.clone();
                                    //let model = Arc::clone(&self.crepe_model);

                                    self.current_stream = Some(self.available_input_devices[i].build_input_stream(
                                        &CONFIG,
                                        move |data: &[i16], info| {
                                            let instant = info.timestamp().callback;

                                            let mut audio_state = cloned_arc.write().unwrap();
                                            if let None = audio_state.first_audio_instant {
                                                audio_state.first_audio_instant = Some(instant);
                                                println!("Updated first audio timestamp");
                                            }

                                            let max_val = data.iter().max().copied().unwrap_or(0);
                                            //let prediction = model.predict_single([0.0; 1024]);
                                            let since_start = instant.duration_since(&audio_state.first_audio_instant.unwrap()).unwrap_or(Duration::ZERO);

                                            audio_state.pitch_points.push([since_start.as_secs_f64(), max_val as f64]);
                                            println!("Data length: {}, max value: {}, since start: {}", data.len(), max_val, since_start.as_secs_f32());
                                            // Explicitly trigger repaint since this thread otherwise is so high-priority that it
                                            // keeps on blocking the render thread through synchronization most of the time.
                                            cloned_ctx.request_repaint();
                                        },
                                        move |err| {
                                            println!("Error: {:?}", err);
                                        },
                                        None,
                                    ).expect("Failed to create input stream!"));
                                    // TODO: call play on stream.
                                }
                            }
                        });
                    if ui.button("Reload devices").clicked() {
                        self.available_input_devices = cpal::default_host().input_devices().expect("Failed to get input devices").collect();
                    }
                    ui.add_space(20.0);

                    ui.add(egui::Slider::new(&mut self.min_display_frequency, 0..=500).text("Min display"));
                    ui.add(egui::Slider::new(&mut self.max_display_frequency, 0..=500).text("Max display"));
                    ui.add(egui::Slider::new(&mut self.min_target_frequency, 0..=500).text("Min target"));
                    ui.add(egui::Slider::new(&mut self.max_target_frequency, 0..=500).text("Max target"));
                });
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.horizontal(|ui| {
                let pin_button = ui.add_sized([40.0, 40.0], egui::ImageButton::new(egui::include_image!("../assets/icons/pin.png")));
                // Right-align settings button by filling remaining space.
                ui.add_space(ui.available_width() - 40.0);
                let settings_button = ui.add_sized([40.0, 40.0], egui::ImageButton::new(egui::include_image!("../assets/icons/settings.png")));

                if pin_button.clicked() {
                    // TODO: Set window level depending on whether button is clicked
                }
                if settings_button.clicked() {
                    self.settings_open = true;
                }
            });
            ui.heading("my cool egui application");
            ui.horizontal(|ui| {
                let name_label = ui.label("Your name: ");
                ui.text_edit_singleline(&mut self.name).labelled_by(name_label.id);
            });
            ui.add(egui::Slider::new(&mut self.age, 0..=120).text("age"));
            if ui.button("Increment").clicked() {
                self.age += 1;
            }
            ui.label(format!("Hello {}, age {}", self.name, self.age));

            let plot = Plot::new("My plot")
                .allow_zoom(false)
                .allow_scroll(false)
                .allow_drag(false)
                .allow_double_click_reset(false);
            let cloned_arc = Arc::clone(&self.audio_state);
            let response = plot.show(ui, move |plot_ui| {
                let target_range_width = (self.max_target_frequency - self.min_target_frequency) as f64;
                let middle_y = self.min_target_frequency as f64 + target_range_width / 2.0;
                plot_ui.hline(HLine::new(middle_y)
                    .width(target_range_width as f32)
                    .color(Color32::PURPLE)
                );
                let audio_state = cloned_arc.read().unwrap();
                let current_secs = if let Some(point) = audio_state.pitch_points.last() { 
                    point[0] 
                } else {
                    10.0
                };
                plot_ui.set_plot_bounds(PlotBounds::from_min_max(
                    [current_secs - 10.0, self.min_display_frequency as f64],
                    [current_secs, self.max_display_frequency as f64],
                ));
                plot_ui.line(Line::new(PlotPoints::new(cloned_arc.read().unwrap().pitch_points.clone())));
                plot_ui.line(Line::new(PlotPoints::from_ys_f32(&[1.0, 3.0, 2.0])));
                plot_ui.line(Line::new(PlotPoints::new(vec![[4.0, 4.0], [5.0, 5.0], [6.0, 6.0]])))
            });
            // Place label over the created plot.
            let rect = response.response.rect;
            // TODO: use actual calculated pitch.
            let text = RichText::new(format!("{}Hz", 123)).size(30.0);
            let label = Label::new(text);
            ui.put(rect, label);
        });
    }
}