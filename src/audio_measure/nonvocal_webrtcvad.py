import webrtcvad
import wave
from pydub import AudioSegment
import io
import os
import csv

def read_wave(path):
    sound = AudioSegment.from_file(path)
    sample_rate = sound.frame_rate

    if sample_rate not in (8000, 16000, 32000, 48000):
        nearest_rate = min((8000, 16000, 32000, 48000), key=lambda x: abs(x-sample_rate))
        print(f"Resampling from {sample_rate} Hz to {nearest_rate} Hz.")
        sound = sound.set_frame_rate(nearest_rate)
        sample_rate = nearest_rate

    if sound.channels != 1:
        sound = sound.set_channels(1)

    audio_data = sound.raw_data

    return audio_data, sample_rate


def frame_generator(frame_duration_ms, audio, sample_rate):
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n < len(audio):
        yield audio[offset:offset + n], timestamp, duration
        timestamp += duration
        offset += n

def calculate_non_vocal_duration(audio_path, frame_duration_ms):
    audio, sample_rate = read_wave(audio_path)
    vad = webrtcvad.Vad()
    
    total_non_vocal_duration = 0
    for frame, timestamp, duration in frame_generator(frame_duration_ms, audio, sample_rate):
        if not vad.is_speech(frame, sample_rate):
            total_non_vocal_duration += duration

    return total_non_vocal_duration

def process_directory(directory, group, frame_duration_ms):
    results = []
    for filename in os.listdir(directory):
        if filename.endswith(".wav"):
            audio_path = os.path.join(directory, filename)
            non_vocal_duration = calculate_non_vocal_duration(audio_path, frame_duration_ms)
            results.append({'group': group, 'filename': filename, 'nonvocal_duration': non_vocal_duration})
    return results

def main():
    frame_duration_ms = 10 
    aphasia_results = process_directory('Aphasia_segment/', 'aphasia', frame_duration_ms)
    control_results = process_directory('Control_segment/', 'control', frame_duration_ms)

    with open('nonvocal_duration_webrtcvad.csv', 'w', newline='') as csvfile:
        fieldnames = ['group', 'filename', 'nonvocal_duration']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for row in aphasia_results + control_results:
            writer.writerow(row)

if __name__ == "__main__":
    main()