//! Implements the CREPE neural-network-based pitch tracker.
//! This code is based on and adapted from the original at https://github.com/marl/crepe, most
//! of it from `crepe/core.py`.
//! It has been modified to only keep the code necessary for this implementation's use cases.

use lazy_static::lazy_static;
use ndarray::Array;
use ort::inputs;
use ort::session::{Session, SessionOutputs};
use std::convert::TryInto;
use std::iter::Iterator;

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
    use float_cmp::assert_approx_eq;

    #[test]
    fn test_cents_mapping() {
        // Values taken as calculated by Python code.
        assert_approx_eq!(f32, CENTS_MAPPING[0], 1997.37940844);
        assert_approx_eq!(f32, CENTS_MAPPING[1], 2017.37940844);
        assert_approx_eq!(f32, CENTS_MAPPING[358], 9157.37940844);
        assert_approx_eq!(f32, CENTS_MAPPING[359], 9177.37940844);
    }

    /// Tests that the frequency and confidence predictions made by the Rust implementation
    /// closely match those of the Python implementation for the given sample .wav files in the
    /// "test-data/" directory.
    #[test]
    fn test_predict_single() -> Result<(), Box<dyn std::error::Error>> {
        ort::init().commit()?;
        let session = Session::builder()?.commit_from_file(ONNX_MODEL_PATH)?;
        let crepe_model = CrepeModel::new(session);

        let wav_files: Vec<std::path::PathBuf> = std::fs::read_dir("test-data")?
            .into_iter()
            .filter(|r| r.is_ok())
            .map(|r| r.unwrap().path())
            .filter(|r| r.is_file() && r.file_name().unwrap().to_string_lossy().ends_with(".wav"))
            .collect();
        for file in wav_files {
            do_test_predict_single(&crepe_model, &file)?;
        }

        Ok(())
    }

    fn do_test_predict_single(
        crepe_model: &CrepeModel,
        wav_file: &std::path::PathBuf,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let parent_dir = wav_file.parent().unwrap().to_string_lossy();
        let stem = wav_file.file_stem().unwrap().to_string_lossy();
        let sample_bytes = std::fs::read(format!("{}/{}_samples.npy", parent_dir, stem))?;
        let frequency_bytes = std::fs::read(format!("{}/{}_frequencies.npy", parent_dir, stem))?;
        let confidence_bytes = std::fs::read(format!("{}/{}_confidences.npy", parent_dir, stem))?;

        let sample_npy = npyz::NpyFile::new(&sample_bytes[..])?;
        let frequency_npy = npyz::NpyFile::new(&frequency_bytes[..])?;
        let confidence_npy = npyz::NpyFile::new(&confidence_bytes[..])?;

        let sample_data = sample_npy.into_vec::<f32>()?;
        // Frequency is output as f64 for some reason, while the rest is f32.
        let frequency_data = frequency_npy.into_vec::<f64>()?;
        let confidence_data = confidence_npy.into_vec::<f32>()?;

        // We only take full 1024-sample frames, so we discard the fractional part.
        let frame_count = (sample_data.len() as f32 / SAMPLES_PER_STEP as f32).floor() as usize;
        assert_eq!(frame_count, frequency_data.len());
        assert_eq!(frame_count, confidence_data.len());

        const U16_MAX: f32 = u16::MAX as f32;
        let i16_sample_chunks: Vec<[i16; SAMPLES_PER_STEP]> = sample_data
            .chunks_exact(SAMPLES_PER_STEP)
            .map(|chunk| {
                chunk
                    .iter()
                    .map(|sample| ((U16_MAX / 2.0) * sample) as i16)
                    .collect::<Vec<i16>>()
                    .try_into()
                    .unwrap()
            })
            .collect::<Vec<[i16; SAMPLES_PER_STEP]>>();
        for (i, sample_chunk) in i16_sample_chunks.iter().enumerate() {
            let prediction = crepe_model.predict_single(*sample_chunk);

            // A rather large epsilon, but fine for our use case.
            const EPSILON: f32 = 0.01;
            assert_approx_eq!(
                f32,
                prediction.frequency,
                frequency_data[i] as f32,
                epsilon = EPSILON
            );
            assert_approx_eq!(
                f32,
                prediction.confidence,
                confidence_data[i],
                epsilon = EPSILON
            );
        }

        Ok(())
    }
}
