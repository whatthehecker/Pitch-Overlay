mod crepe;
mod app;

use crate::app::{PitchOverlayApp, Settings, SETTINGS_STORAGE_KEY};
use crate::crepe::CrepeModel;
use cpal::traits::HostTrait;
use cpal::Device;
use eframe::{egui, CreationContext};
use ort::session::Session;

const ONNX_MODEL_PATH: &str = "crepe-full.onnx";


fn read_stored_settings(cc: &CreationContext) -> Option<Settings> {
    cc.storage?.get_string(SETTINGS_STORAGE_KEY)
        .map(|value| serde_json::from_str(value.as_str()))?
        .map_or(None, |settings| Some(settings))
}

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

            let settings = read_stored_settings(cc).unwrap_or(Settings::default());

            Ok(Box::<PitchOverlayApp>::new(PitchOverlayApp::new(
                all_devices,
                crepe_model,
                settings,
            )))
        }),
    )
}
