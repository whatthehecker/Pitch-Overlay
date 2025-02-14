use ndarray::{Array};
use ort::inputs;
use ort::session::{Session, SessionOutputs};

#[derive(Debug)]
pub struct Prediction {
    pub frequency: f32,
    pub confidence: f32,
}

pub const SAMPLES_PER_STEP: usize = 1024;

type Activation = [f32; 360];
type Cents = [f32; 1];

pub struct CrepeModel {
    model: Session,
}

impl CrepeModel {
    pub fn new(model: Session) -> Self {
        CrepeModel {
            model
        }
    }

    fn get_activation(&self, audio: [f32; SAMPLES_PER_STEP]) -> Activation {
        /*
        model = build_and_load_model(model_capacity)

        if len(audio.shape) == 2:
            audio = audio.mean(1)  # make mono
        audio = audio.astype(np.float32)
        if sr != model_srate:
        # resample audio if necessary
        from resampy import resample
        audio = resample(audio, sr, model_srate)

        # pad so that frames are centered around their timestamps (i.e. first frame
                                                                   # is zero centered).
        if center:
            audio = np.pad(audio, 512, mode='constant', constant_values=0)

        # make 1024-sample frames of the audio with hop length of 10 milliseconds
        hop_length = int(model_srate * step_size / 1000)
        n_frames = 1 + int((len(audio) - 1024) / hop_length)
        frames = as_strided(audio, shape=(1024, n_frames),
                            strides=(audio.itemsize, hop_length * audio.itemsize))
        frames = frames.transpose().copy()

        # normalize each frame -- this is expected by the model
        frames -= np.mean(frames, axis=1)[:, np.newaxis]
        frames /= np.clip(np.std(frames, axis=1)[:, np.newaxis], 1e-8, None)

        # run prediction and convert the frequency bin weights to Hz
        return model.predict(frames, verbose=verbose)
         */

        let input= Array::from_iter(audio);
        let outputs: SessionOutputs = self.model.run(inputs!["input" => input.view()].unwrap()).unwrap();
        let output = outputs["output_0"].try_extract_tensor::<f32>().unwrap();

        output.as_slice().unwrap().try_into().unwrap()
    }

    fn to_local_average_cents(&self, activation: Activation) -> Cents {
        return [0.0; 1];
    }

    pub fn predict_single(&self, audio: [f32; SAMPLES_PER_STEP]) -> Prediction {
        let activation = self.get_activation(audio);
        let confidence = activation.into_iter().reduce(f32::max).unwrap_or(0.0);
        let cents = self.to_local_average_cents(activation);

        let frequency = cents.map(|x| 10.0 * 2.0_f32.powf(x / 1200.0));
        // frequency[np.isnan(frequency)] = 0

        Prediction {
            frequency: frequency[0],
            confidence,
        }
    }
}

