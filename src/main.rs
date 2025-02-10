use eframe::{egui, Frame};
use eframe::egui::{Context};

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([320.0, 240.0]),
        ..Default::default()
    };
    eframe::run_native(
        "My egui App",
        options,
        Box::new(|cc| {
            egui_extras::install_image_loaders(&cc.egui_ctx);

            Ok(Box::<MyApp>::default())
        }),
    )
}

struct MyApp {
    name: String,
    age: u32,
    settings_open: bool,
}

impl Default for MyApp {
    fn default() -> Self {
        Self {
            name: "Arthur".to_owned(),
            age: 42,
            settings_open: false,
        }
    }
}

impl eframe::App for MyApp {
    fn update(&mut self, ctx: &Context, _frame: &mut Frame) {
        if self.settings_open {
            egui::Window::new("Settings")
                .collapsible(false)
                .open(&mut self.settings_open)
                .show(ctx, |ui| {
                    ui.label("Test content");
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
        });
    }
}