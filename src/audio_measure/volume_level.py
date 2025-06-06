import os
import librosa
import numpy as np
import csv

def calculate_volume_level(file_path):

    audio, sr = librosa.load(file_path)
    rms = np.sqrt(np.mean(audio**2))
    return rms

def process_directory(directory_path, group):

    all_files = os.listdir(directory_path)
    audio_files = [file for file in all_files if file.endswith('.wav')]

    results = []
    for file in audio_files:
        full_path = os.path.join(directory_path, file)
        volume_level = calculate_volume_level(full_path)
        results.append([file, group, volume_level])

    return results

def main():
    directories = {'Aphasia_segment/': 'aphasia', 'Control_segment/': 'control'}

    all_results = []
    for directory, group in directories.items():
        results = process_directory(directory, group)
        all_results.extend(results)

    with open('../../data/Demographic/volume_levels.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['filename', 'group', 'volume_level'])
        writer.writerows(all_results)

    print("CSV file created: volume_levels.csv")

main()
