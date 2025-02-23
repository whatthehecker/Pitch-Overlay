mod crepe;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{BufferSize, Device, SampleRate, Stream, StreamConfig, StreamInstant};
use eframe::egui::{Color32, Context, Label, Rgba, RichText, ViewportCommand, WindowLevel};
use eframe::{egui, Frame, Storage};
use egui_plot::{HLine, Line, Plot, PlotBounds, PlotPoints};
use std::sync::{Arc, RwLock};
use std::time::Duration;
use eframe::egui::color_picker::Alpha;
use ort::session::Session;
use serde::{Deserialize, Serialize};
use crate::crepe::{CrepeModel};

const SETTINGS_STORAGE_KEY: &str = "settings";
const ONNX_MODEL_PATH: &str = "crepe-full.onnx";

/// The number of CREPE predictions to combine into a single averaged pitch value.
///
/// By default, CREPE takes 64 millis of audio which results in really fast predictions that are all
/// over the place.
/// To display anything useful, aggregate a multiple of 64 milliseconds of audio, run pitch prediction
/// on each 64 millis chunk and average the values.
const STEPS_PER_DISPLAY: usize = 2;

const MIN_SAMPLES_PER_DISPLAY: usize = STEPS_PER_DISPLAY * crepe::SAMPLES_PER_STEP;

const CONFIG: StreamConfig = StreamConfig {
    channels: 1,
    sample_rate: SampleRate(crepe::SAMPLE_RATE),
    buffer_size: BufferSize::Fixed(MIN_SAMPLES_PER_DISPLAY as u32),
};

fn main() -> eframe::Result {
    ort::init()
        .commit()
        .expect("Failed to init ort.");
    let session = Session::builder()
        .expect("Failed to create ONNX session.")
        .commit_from_file(ONNX_MODEL_PATH)
        .expect(format!("Failed to find model file at \"{}\"", ONNX_MODEL_PATH).as_str());
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
                }
                None => Settings::default(),
            };

            Ok(Box::<PitchOverlayApp>::new(PitchOverlayApp::new(
                all_devices,
                crepe_model,
                stored_settings,
            )))
        }),
    )
}

struct WindowState {
    is_always_on_top: bool,
    are_settings_open: bool,
}

impl Default for WindowState {
    fn default() -> Self {
        WindowState {
            is_always_on_top: false,
            are_settings_open: false,
        }
    }
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
    label_color: Rgba,
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
            label_color: Rgba::from(Color32::WHITE),
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

struct PitchOverlayApp {
    current_stream: Option<Stream>,
    current_device_index: Option<usize>,
    available_input_devices: Vec<Device>,

    audio_state: Arc<RwLock<AudioState>>,
    crepe_model: Arc<CrepeModel>,
    settings: Settings,

    window_state: WindowState,
}

impl PitchOverlayApp {
    fn new(input_devices: Vec<Device>, crepe_model: CrepeModel, settings: Settings) -> Self {
        Self {
            current_stream: None,
            current_device_index: None,
            available_input_devices: input_devices,

            audio_state: Arc::new(RwLock::new(AudioState::default())),
            crepe_model: Arc::new(crepe_model),
            settings,

            window_state: WindowState::default(),
        }
    }

    fn current_device(&self) -> Option<&Device> {
        if let Some(i) = self.current_device_index {
            return Some(&self.available_input_devices[i]);
        }

        None
    }
}

impl eframe::App for PitchOverlayApp {
    fn update(&mut self, ctx: &Context, _frame: &mut Frame) {
        if self.window_state.are_settings_open {
            egui::Window::new("Settings")
                .collapsible(false)
                .open(&mut self.window_state.are_settings_open)
                .show(ctx, |ui| {
                    ui.add(egui::Slider::new(&mut self.settings.confidence_threshold, 0.0..=1.0).text("Pitch confidence threshold"));
                    ui.add_space(20.0);

                    ui.horizontal(|ui| {
                        // TODO: right-align these, looks weird
                        egui::widgets::color_picker::color_edit_button_rgba(ui, &mut self.settings.target_color, Alpha::BlendOrAdditive);
                        ui.add(Label::new("Target region color")).on_hover_ui(|ui| {
                            ui.label("Color of the target region on the plot");
                        });
                    });
                    ui.horizontal(|ui| {
                        egui::widgets::color_picker::color_edit_button_rgba(ui, &mut self.settings.label_color, Alpha::BlendOrAdditive);
                        ui.add(Label::new("Pitch label color")).on_hover_ui(|ui| {
                            ui.label("Color in the center of the plot of the label that displays your current pitch");
                        })
                    });
                    ui.add_space(20.0);

                    let min_display_response = ui.add(egui::Slider::new(&mut self.settings.display_range.0, 0..=499).text("Min display")).on_hover_ui(|ui| {
                        ui.label("Minimum frequency to display on the graph.");
                    });
                    let max_display_response = ui.add(egui::Slider::new(&mut self.settings.display_range.1, 1..=500).text("Max display")).on_hover_ui(|ui| {
                        ui.label("Maximum frequency to display on the graph");
                    });
                    let display_range_changed = min_display_response.changed()
                        | max_display_response.changed();
                    if display_range_changed {
                        let (lower, upper) = &mut self.settings.display_range;
                        if lower >= upper {
                            *upper = *lower + 1;
                        }
                    }

                    // TODO: these only show tooltips when the slider itself is hovered, while the color setting shows its tooltip when the label is hovered, that's inconsistent.
                    let min_target_response = ui.add(egui::Slider::new(&mut self.settings.target_range.0, 0..=499).text("Min target")).on_hover_ui(|ui| {
                        ui.label("Minimum frequency you are aiming for");
                    });
                    let max_target_response = ui.add(egui::Slider::new(&mut self.settings.target_range.1, 1..=500).text("Max target")).on_hover_ui(|ui| {
                        ui.label("Maximum frequency you are aiming for");
                    });
                    let target_range_changed = min_target_response.changed()
                        | max_target_response.changed();
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
            let current_device_name = self.current_device().map(|device| device.name().unwrap_or("Unnamed device".to_owned())).unwrap_or("Audio disconnected".to_owned());

            ui.horizontal_wrapped(|ui| {
                egui::ComboBox::from_id_salt("Audio Input device")
                    .truncate()
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

                                match self.available_input_devices[i].build_input_stream(
                                    &CONFIG,
                                    move |data: &[i16], info| {
                                        let instant = info.timestamp().callback;

                                        let mut audio_state = cloned_arc.write().unwrap();
                                        if let None = audio_state.first_audio_instant {
                                            audio_state.first_audio_instant = Some(instant);
                                            println!("Updated first audio timestamp");
                                        }

                                        audio_state.recent_audio.extend_from_slice(data);

                                        let sample_count = audio_state.recent_audio.len();
                                        if sample_count < MIN_SAMPLES_PER_DISPLAY {
                                            return;
                                        }

                                        let most_recent_audio: [i16; MIN_SAMPLES_PER_DISPLAY] = (&audio_state.recent_audio[sample_count - MIN_SAMPLES_PER_DISPLAY..sample_count]).try_into().unwrap();
                                        let predictions = most_recent_audio.chunks_exact(crepe::SAMPLES_PER_STEP)
                                            .map(|chunk| model.predict_single(chunk.try_into().unwrap()))
                                            .filter(|prediction|
                                                prediction.confidence >= settings.confidence_threshold
                                                    && prediction.frequency >= settings.display_range.0 as f32
                                                    && prediction.frequency <= settings.display_range.1 as f32)
                                            .map(|prediction| prediction.frequency)
                                            .collect::<Vec<f32>>();
                                        let average_pitch = if predictions.is_empty() {
                                            f32::NAN
                                        } else {
                                            predictions.iter().sum::<f32>() / predictions.len() as f32
                                        };
                                        audio_state.recent_audio.clear();

                                        let since_start = instant.duration_since(&audio_state.first_audio_instant.unwrap()).unwrap_or(Duration::ZERO);
                                        audio_state.pitch_points.push([since_start.as_secs_f64(), average_pitch as f64]);
                                        if !average_pitch.is_nan() {
                                            audio_state.last_valid_frequency = Some(average_pitch);
                                        }

                                        // Explicitly trigger repaint since this thread otherwise is so high-priority that it
                                        // keeps on blocking the render thread through synchronization most of the time.
                                        cloned_ctx.request_repaint();
                                    },
                                    move |err| {
                                        println!("Error: {:?}", err);
                                    },
                                    None,
                                ) {
                                    Err(e) => {
                                        self.current_stream = None;
                                        self.current_device_index = None;
                                        // TODO: display error in UI.
                                    }
                                    Ok(stream) => {
                                        match stream.play() {
                                            Err(e) => {
                                                // TODO: display error in UI.
                                                self.current_stream = None;
                                                self.current_device_index = None;
                                            },
                                            Ok(_) => self.current_stream = Some(stream),
                                        }
                                    }
                                };
                            }
                        }
                    });
                if ui.button("Reload devices").clicked() {
                    self.available_input_devices = cpal::default_host().input_devices().expect("Failed to get input devices").collect();
                }

                let checkbox_changed = ui.add_sized([80.0, 20.0], egui::Checkbox::new(&mut self.window_state.is_always_on_top, "Always on top")).changed();
                let settings_button = ui.add_sized([100.0, 20.0], egui::Button::new("Settings"));

                if checkbox_changed {
                    let new_level = if self.window_state.is_always_on_top {
                        WindowLevel::AlwaysOnTop
                    } else {
                        WindowLevel::Normal
                    };
                    ctx.send_viewport_cmd(ViewportCommand::WindowLevel(new_level))
                }
                if settings_button.clicked() {
                    self.window_state.are_settings_open = true;
                }
            });

            let current_device_index = self.current_device_index;
            let label_color = self.settings.label_color;
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
                None => match current_device_index {
                    None => "No device selected.",
                    Some(_) => "Waiting for audio data...",
                }.to_owned(),
                Some(frequency) => format!("{}Hz", frequency as u32),
            };
            let text = RichText::new(display_frequency).size(30.0).color(label_color);
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
            }
            Err(e) => println!("Error saving settings: {}", e),
        }
    }
}