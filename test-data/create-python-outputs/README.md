The Python scripts inside this directory take the audio files from the parent directory, writing the raw samples as 
binary files and the corresponding output by the Python CREPE implementation to files.
These can then later be used to check whether the Rust implementation generates equal (or at least similar) inference
outputs compared to the original implementation in Python.