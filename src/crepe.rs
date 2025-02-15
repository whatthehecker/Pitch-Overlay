use std::convert::TryInto;
use std::iter::Iterator;
use ndarray::{Array};
use ort::inputs;
use ort::session::{Session, SessionOutputs};

// TODO: document that this code is adapted from the official CREPE Python package 
// TODO: add license that is compatible with CREPE's license

#[derive(Debug)]
pub struct Prediction {
    pub frequency: f32,
    pub confidence: f32,
}

pub const SAMPLES_PER_STEP: usize = 1024;

type Activation = [f32; 360];

fn argmax(values: &[f32]) -> Option<usize> {
    values.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(i, _)| i)
}

fn mean(values: &[f32]) -> f32 {
    values.iter().sum::<f32>() / values.len() as f32
}

fn std(values: &[f32]) -> f32 {
    let mean = mean(values);
    let variance = values.iter().map(|value| {
        let diff = mean - *value;

        diff * diff
    }).sum::<f32>() / values.len() as f32;

    variance.sqrt()
}

pub struct CrepeModel {
    model: Session,
}

impl CrepeModel {
    pub fn new(model: Session) -> Self {
        CrepeModel {
            model
        }
    }

    fn get_activation(&self, audio: [i16; SAMPLES_PER_STEP]) -> Activation {
        let audio = audio.map(|x| x as f32);
        // Pad audio with 512 zeros from either side.
        // TODO: check whether this is actually needed.
        //let mut centered_audio = [0.0; 512 + 1024 + 512];
        //centered_audio[512..(512 + 1024)].copy_from_slice(audio.as_slice());
        let mean = mean(&audio);
        let std = std(&audio);
        let clipped_std = std.clamp(1e-8, f32::MAX);
        let normalized_audio = audio.map(|x| (x - mean) / clipped_std);

        let input= Array::from_iter(normalized_audio).into_shape_with_order((1, 1024)).unwrap();
        let outputs: SessionOutputs = self.model.run(inputs!["input" => input.view()].unwrap()).unwrap();
        let output = outputs["output_0"].try_extract_tensor::<f32>().unwrap();

        output.as_slice().unwrap().try_into().unwrap()
    }

    fn to_local_average_cents(&self, activation: Activation) -> f32 {
        // TODO: needs testing whether this results in the correct factors compared to Python source.
        let cents_mapping: [f32; 360] = (0..360)
            .map(|x| x as f32 * (7180.0 / 360.0) + 1997.3794084376191)
            .collect::<Vec<f32>>()
            .try_into()
            .unwrap();

        let center = argmax(&activation).unwrap();

        let start = center.saturating_sub(4);
        let end = 1024.min(center + 5);
        // TODO: this is still somewhat wrong, sometimes i == 360 but len == 360.
        let product_sum: f32 = (start..end).map(|i| activation[i] * cents_mapping[i]).sum();
        let weight_sum: f32 = activation.iter().sum();

        product_sum / weight_sum
    }

    pub fn predict_single(&self, audio: [i16; SAMPLES_PER_STEP]) -> Prediction {
        let activation = self.get_activation(audio);
        let confidence = activation.into_iter().reduce(f32::max).unwrap_or(0.0);
        let cents = self.to_local_average_cents(activation);
        let frequency = 10.0 * 2.0_f32.powf(cents / 1200.0);

        Prediction {
            frequency,
            confidence,
        }
    }
}

