# calculate noise level using librosa threshold

import os
import pandas as pd
import librosa
import numpy as np

def calculate_background_noise(audio, frame_size=2048, hop_size=512, threshold=0.01):
    energy = librosa.feature.rms(y=audio, frame_length=frame_size, hop_length=hop_size)[0]
    non_speech_frames = energy < threshold
    
    background_noise_segments = []
    current_segment = []
    
    for i, is_non_speech in enumerate(non_speech_frames):
        if is_non_speech:
            current_segment.append(i)
        elif current_segment:
            background_noise_segments.append(current_segment)
            current_segment = []
    
    if current_segment:
        background_noise_segments.append(current_segment)
    
    background_noise_levels = []
    
    for segment in background_noise_segments:
        segment_audio = audio[segment[0] * hop_size : (segment[-1] + 1) * hop_size]
        segment_energy = librosa.feature.rms(y=segment_audio, frame_length=frame_size, hop_length=hop_size)[0]
        background_noise_levels.append(np.mean(segment_energy))
    
    return background_noise_levels

if __name__ == "__main__":
    input_directory = "Aphasia_segment/"
    output_csv = "Aphasia_noiselevel.csv"
    
    audio_files = [file for file in os.listdir(input_directory) if file.endswith(".wav")]
    
    data = []
    
    for audio_file in audio_files:
        audio_path = os.path.join(input_directory, audio_file)
        audio, _ = librosa.load(audio_path, sr=16000)
        background_noise_levels = calculate_background_noise(audio)
        
        if background_noise_levels:
            mean_background_noise = np.mean(background_noise_levels)
            data.append({"File Name": audio_file, "Mean Background Noise": mean_background_noise})
        else:
            data.append({"File Name": audio_file, "Mean Background Noise": np.nan})
    
    df = pd.DataFrame(data)
    
    df.to_csv(output_csv, index=False)
