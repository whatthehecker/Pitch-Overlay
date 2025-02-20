mod crepe;

use cpal::traits::{DeviceTrait, HostTrait};
use cpal::{BufferSize, Device, SampleRate, Stream, StreamConfig, StreamInstant};
use eframe::egui::{Color32, Context, Label, Rgba, RichText};
use eframe::{egui, Frame, Storage};
use egui_plot::{HLine, Line, Plot, PlotBounds, PlotPoints};
use std::sync::{Arc, RwLock};
use std::time::Duration;
use eframe::egui::color_picker::Alpha;
use ort::session::Session;
use serde::{Deserialize, Serialize};
use crate::crepe::{CrepeModel, SAMPLES_PER_STEP};

const CONFIG: StreamConfig = StreamConfig {
    channels: 1,
    sample_rate: SampleRate(16_000),
    // 1024 is 64 milliseconds, the smallest unit that CREPE can handle.
    buffer_size: BufferSize::Fixed(1024),
};
const SETTINGS_STORAGE_KEY: &str = "settings";

fn main() -> eframe::Result {
    ort::init()
        .commit()
        .expect("Failed to init ort.");
    let session = Session::builder().unwrap()
        .commit_from_file("assets/crepe-full.onnx").unwrap();
    let crepe_model = CrepeModel::new(session);

    let host = cpal::default_host();
    let all_devices = host.input_devices()
        .expect("Failed to get input devices")
        .map(|device| device.clone())
        .collect::<Vec<Device>>();
    
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([320.0, 240.0]),
        ..Default::default()
    };
    eframe::run_native(
        "Pitch Overlay",
        options,
        Box::new(|cc| {
            egui_extras::install_image_loaders(&cc.egui_ctx);
            
            let stored_settings = match cc.storage {
                Some(storage) => {
                    match storage.get_string(SETTINGS_STORAGE_KEY) {
                        Some(content) => serde_json::from_str(content.as_str()).unwrap_or(Settings::default()),
                        None => Settings::default(),
                    }
                },
                None => Settings::default(),
            };

            Ok(Box::<MyApp>::new(MyApp::new(
                all_devices,
                crepe_model,
                stored_settings,
            )))
        }),
    )
}

struct AudioState {
    first_audio_instant: Option<StreamInstant>,
    // Temporary store for any audio data that was less than 1024 samples long.
    // Some audio backends output less than 1024 samples per callback, so we need to aggregate
    // some values until we have those 1024 entries.
    recent_audio: Vec<i16>,
    last_valid_frequency: Option<f32>,
    pitch_points: Vec<[f64; 2]>,
}

/// Settings of the application which are persisted between sessions.
#[derive(Debug, Serialize, Deserialize, Clone, Copy)]
struct Settings {
    display_range: (u32, u32),
    target_range: (u32, u32),
    confidence_threshold: f32,
    target_color: Rgba,
    // TODO: add configurable color of target region
    // TODO: uncomment and implement restoring last device on open if selected
    //restore_last_device: bool,
    //last_device_id: ???
}

impl Default for Settings {
    fn default() -> Self {
        Settings {
            display_range: (50, 500),
            target_range: (185, 300),
            confidence_threshold: 0.5,
            target_color: Rgba::from(Color32::LIGHT_GREEN),
        }
    }
}

impl Default for AudioState {
    fn default() -> Self {
        AudioState {
            first_audio_instant: None,
            recent_audio: vec![],
            last_valid_frequency: None,
            pitch_points: vec![],
        }
    }
}

struct MyApp {
    current_stream: Option<Stream>,
    current_device_index: Option<usize>,
    available_input_devices: Vec<Device>,

    audio_state: Arc<RwLock<AudioState>>,
    crepe_model: Arc<CrepeModel>,
    settings: Settings,

    settings_open: bool,
}

impl MyApp {
    fn new(input_devices: Vec<Device>, crepe_model: CrepeModel, settings: Settings) -> Self {
        Self {
            current_stream: None,
            current_device_index: None,
            available_input_devices: input_devices,

            audio_state: Arc::new(RwLock::new(AudioState::default())),
            crepe_model: Arc::new(crepe_model),
            settings,

            settings_open: false,
        }
    }

    fn current_device(&self) -> Option<&Device> {
        if let Some(i) = self.current_device_index {
            return Some(&self.available_input_devices[i]);
        }

        None
    }
}

impl eframe::App for MyApp {
    fn update(&mut self, ctx: &Context, _frame: &mut Frame) {
        // TODO: move device selection from settings onto main screen (maybe hide it if something is selected until hovering over the window though)
        
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
                                    let model = Arc::clone(&self.crepe_model);
                                    
                                    let settings = self.settings;

                                    self.current_stream = Some(self.available_input_devices[i].build_input_stream(
                                        &CONFIG,
                                        move |data: &[i16], info| {
                                            let instant = info.timestamp().callback;

                                            let mut audio_state = cloned_arc.write().unwrap();
                                            if let None = audio_state.first_audio_instant {
                                                audio_state.first_audio_instant = Some(instant);
                                                println!("Updated first audio timestamp");
                                            }

                                            // Aggregate audio samples, then calculate new pitch if
                                            // at least 1024 samples are now in the buffer.
                                            audio_state.recent_audio.extend_from_slice(data);

                                            let sample_count = audio_state.recent_audio.len();
                                            if sample_count < SAMPLES_PER_STEP {
                                                return;
                                            }
                                            
                                            let most_recent_audio: [i16; SAMPLES_PER_STEP] = (&audio_state.recent_audio[sample_count - SAMPLES_PER_STEP..sample_count]).try_into().unwrap();
                                            let prediction = model.predict_single(most_recent_audio);
                                            audio_state.recent_audio.clear();
                                            
                                            let effective_frequency = if prediction.confidence >= settings.confidence_threshold { 
                                                prediction.frequency
                                            } else { 
                                                f32::NAN 
                                            };
                                            let since_start = instant.duration_since(&audio_state.first_audio_instant.unwrap()).unwrap_or(Duration::ZERO);
                                            audio_state.pitch_points.push([since_start.as_secs_f64(), effective_frequency as f64]);
                                            if !effective_frequency.is_nan() {
                                                audio_state.last_valid_frequency = Some(effective_frequency);
                                            }
                                            
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
                    
                    ui.add(egui::Slider::new(&mut self.settings.confidence_threshold, 0.0..=1.0).text("Pitch confidence threshold"));
                    ui.add_space(20.0);
                    
                    ui.horizontal(|ui| {
                        // TODO: right-align these, looks weird
                        egui::widgets::color_picker::color_edit_button_rgba(ui, &mut self.settings.target_color, Alpha::BlendOrAdditive);
                        ui.add(Label::new("Target region color"));
                    });
                    ui.add_space(20.0);

                    let display_range_changed = ui.add(egui::Slider::new(&mut self.settings.display_range.0, 0..=499).text("Min display")).changed()
                        | ui.add(egui::Slider::new(&mut self.settings.display_range.1, 1..=500).text("Max display")).changed();
                    if display_range_changed {
                        let (lower, upper) = &mut self.settings.display_range;
                        if lower >= upper {
                            *upper = *lower + 1;
                        }
                    }

                    let target_range_changed = ui.add(egui::Slider::new(&mut self.settings.target_range.0, 0..=499).text("Min target")).changed()
                        | ui.add(egui::Slider::new(&mut self.settings.target_range.1, 1..=500).text("Max target")).changed();
                    if target_range_changed {
                        let (lower, upper) = &mut self.settings.target_range;
                        if lower >= upper {
                            *upper = *lower + 1;
                        }
                    }
                });
        }

        let arc1 = Arc::clone(&self.audio_state);
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
            // TODO: maybe display hint to choose audio device if nothing is selected here

            let plot = Plot::new("My plot")
                .allow_zoom(false)
                .allow_scroll(false)
                .allow_drag(false)
                .allow_double_click_reset(false);
            let cloned_arc = Arc::clone(&self.audio_state);
            let response = plot.show(ui, move |plot_ui| {
                let target_range_width = (self.settings.target_range.1 - self.settings.target_range.0) as f64;
                let middle_y = self.settings.target_range.0 as f64 + target_range_width / 2.0;
                plot_ui.hline(HLine::new(middle_y)
                    .width(target_range_width as f32)
                    .color(self.settings.target_color)
                );
                let audio_state = cloned_arc.read().unwrap();
                let current_secs = if let Some(point) = audio_state.pitch_points.last() {
                    point[0]
                } else {
                    10.0
                };
                plot_ui.set_plot_bounds(PlotBounds::from_min_max(
                    [current_secs - 10.0, self.settings.display_range.0 as f64],
                    [current_secs, self.settings.display_range.1 as f64],
                ));
                plot_ui.line(Line::new(PlotPoints::new(cloned_arc.read().unwrap().pitch_points.clone())));
            });
            // Place label over the created plot.
            let rect = response.response.rect;
            let display_frequency = match arc1.read().unwrap().last_valid_frequency {
                None => "xxxHz".to_owned(),
                Some(frequency) => format!("{}Hz", frequency as u32),
            };
            let text = RichText::new(display_frequency).size(30.0);
            // TODO: this needs to be rendered e. g. with a white outline to be visible on all colors. 
            let label = Label::new(text);
            ui.put(rect, label);
        });
    }

    fn save(&mut self, storage: &mut dyn Storage) {
        println!("Saving settings...");
        match serde_json::to_string(&self.settings) {
            Ok(json) => {
                storage.set_string(SETTINGS_STORAGE_KEY, json);
                println!("Saved settings.");
            },
            Err(e) => println!("Error saving settings: {}", e),
        }
    }
}