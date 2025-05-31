SAMPLING_RATE = 16000
import os
import csv
import torch
torch.set_num_threads(1)
USE_ONNX = False

model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                              model='silero_vad',
                              force_reload=True,
                              onnx=USE_ONNX)

(get_speech_timestamps,
 save_audio,
 read_audio,
 VADIterator,
 collect_chunks) = utils


def calculate_nonvocal_duration(speech_timestamps, total_duration, sampling_rate):
    """Calculate the total duration of non-vocal segments."""
    vocal_duration = 0
    for timestamp in speech_timestamps:
        start_sec = timestamp['start'] / sampling_rate
        end_sec = timestamp['end'] / sampling_rate
        vocal_duration += (end_sec - start_sec)

    return total_duration - vocal_duration

directories = {'Aphasia_segment': 'aphasia', 'Control_segment': 'control'}

output_file = 'nonvocal_duration_silero.csv'

with open(output_file, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['group', 'filename', 'nonvocal_duration'])

    for directory, group in directories.items():
        for filename in os.listdir(directory):
            if filename.endswith('.wav'):
                file_path = os.path.join(directory, filename)
                wav = read_audio(file_path, sampling_rate=SAMPLING_RATE)
                speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=SAMPLING_RATE)
                total_duration = len(wav) / SAMPLING_RATE
                nonvocal_duration = calculate_nonvocal_duration(speech_timestamps, total_duration, SAMPLING_RATE)
                writer.writerow([group, filename, nonvocal_duration])
