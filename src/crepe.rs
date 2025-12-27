use lazy_static::lazy_static;
use ndarray::Array;
use ort::inputs;
use ort::session::{Session, SessionOutputs};
use std::convert::TryInto;
use std::iter::Iterator;

// TODO: document that this code is adapted from the official CREPE Python package

/// Outputs of the CREPE model for a single 1024-sample audio chunk.
#[derive(Debug)]
pub struct Prediction {
    pub frequency: f32,
    pub confidence: f32,
}

/// The default audio sample rate that is expected by the CREPE model.
pub const SAMPLE_RATE: u32 = 16_000;

/// The number of samples that is used to predict a single pitch output.
pub const SAMPLES_PER_STEP: usize = 1024;

type Activation = [f32; 360];

fn argmax(values: &[f32]) -> Option<usize> {
    values
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(i, _)| i)
}

fn mean(values: &[f32]) -> f32 {
    values.iter().sum::<f32>() / values.len() as f32
}

fn std(values: &[f32]) -> f32 {
    let mean = mean(values);
    let variance = values
        .iter()
        .map(|value| {
            let diff = mean - *value;

            diff * diff
        })
        .sum::<f32>()
        / values.len() as f32;

    variance.sqrt()
}

pub struct CrepeModel {
    model: Session,
}

lazy_static! {
    static ref CENTS_MAPPING: [f32; 360] = (0..360)
        .map(|x| x as f32 * 20.0 + 1997.3794084376191)
        .collect::<Vec<f32>>()
        .try_into()
        .unwrap();
}

impl CrepeModel {
    pub fn new(model: Session) -> Self {
        CrepeModel { model }
    }

    fn get_activation(&self, audio: [i16; SAMPLES_PER_STEP]) -> Activation {
        let audio = audio.map(|x| x as f32 / i16::MAX as f32);
        // Pad audio with 512 zeros from either side.
        // TODO: check whether this is actually needed.
        //let mut centered_audio = [0.0; 512 + 1024 + 512];
        //centered_audio[512..(512 + 1024)].copy_from_slice(audio.as_slice());
        let mean = mean(&audio);
        let centered_audio = audio.map(|x| x - mean);
        let std = std(&centered_audio);
        let clipped_std = std.clamp(1e-8, f32::MAX);
        let normalized_audio = audio.map(|x| (x - mean) / clipped_std);

        let input = Array::from_iter(normalized_audio)
            .into_shape_with_order((1, 1024))
            .unwrap();
        let outputs: SessionOutputs = self
            .model
            .run(inputs!["input" => input.view()].unwrap())
            .unwrap();
        let output = outputs["output_0"].try_extract_tensor::<f32>().unwrap();

        output.as_slice().unwrap().try_into().unwrap()
    }

    fn to_local_average_cents(&self, activation: Activation) -> f32 {
        let center = argmax(&activation).unwrap();
        let start = center.saturating_sub(4);
        let end = (center + 5).min(activation.len());
        let product_sum: f32 = (start..end).map(|i| activation[i] * CENTS_MAPPING[i]).sum();
        let weight_sum: f32 = (start..end).map(|i| activation[i]).sum();

        product_sum / weight_sum
    }

    /// Calculates the model output for a single audio chunk.
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

#[cfg(test)]
mod tests {
    use crate::crepe::*;
    use crate::ONNX_MODEL_PATH;
    use approx::assert_relative_eq;

    #[test]
    fn test_cents_mapping() {
        // Values taken as calculated by Python code.
        assert_relative_eq!(CENTS_MAPPING[0], 1997.37940844);
        assert_relative_eq!(CENTS_MAPPING[1], 2017.37940844);
        assert_relative_eq!(CENTS_MAPPING[358], 9157.37940844);
        assert_relative_eq!(CENTS_MAPPING[359], 9177.37940844);
    }

    #[test]
    fn test_predict_single() -> Result<(), Box<dyn std::error::Error>> {
        ort::init().commit()?;
        let session = Session::builder()?.commit_from_file(ONNX_MODEL_PATH)?;
        let crepe_model = CrepeModel::new(session);

        let sample_bytes = std::fs::read("test-data/sweep_samples.npy")?;
        let frequency_bytes = std::fs::read("test-data/sweep_frequencies.npy")?;
        let confidence_bytes = std::fs::read("test-data/sweep_confidences.npy")?;

        let sample_npy = npyz::NpyFile::new(&sample_bytes[..])?;
        let frequency_npy = npyz::NpyFile::new(&frequency_bytes[..])?;
        let confidence_npy = npyz::NpyFile::new(&confidence_bytes[..])?;

        let sample_data = sample_npy.into_vec::<f32>()?;
        // Frequency is output as f64 for some reason, while the rest is f32.
        let frequency_data = frequency_npy.into_vec::<f64>()?;
        let confidence_data = confidence_npy.into_vec::<f32>()?;

        const U16_MAX: f32 = u16::MAX as f32;
        // TODO: build chunks of 1024 samples instead of only a single chunk.
        let i16_samples: [i16; SAMPLES_PER_STEP] = sample_data
            .iter()
            .map(|sample| ((U16_MAX / 2.0) * sample) as i16)
            .take(SAMPLES_PER_STEP)
            .collect::<Vec<i16>>()
            .try_into()
            .expect("There should be 1024 samples.");
        let prediction = crepe_model.predict_single(i16_samples);

        assert_relative_eq!(prediction.frequency, frequency_data[0] as f32);
        assert_relative_eq!(prediction.confidence, confidence_data[0]);

        Ok(())
    }

    // TODO: add tests for comparing calculated output of some example audio with Python output.
}
