import os
import numpy as np
import pandas as pd
from scipy.io import wavfile
from pydub import AudioSegment

def add_silence_beginning(input_file, output_file):
    
    silence_duration = 10000  
    silent_segment = AudioSegment.silent(duration=silence_duration)  
    audio_segment = AudioSegment.from_wav(input_file)
    final_audio = silent_segment + audio_segment
    final_audio.export(output_file, format="wav")

def convert_to_mono(stereo_file):
    
    fs, data = wavfile.read(stereo_file)
    if data.ndim == 2:
        mono_data = np.mean(data, axis=1)
    else:
        mono_data = data
    return fs, mono_data.astype(np.float32)

def generate_white_noise(duration_ms, fs, snr_db, signal_power):

    duration_samples = int(fs * (duration_ms / 1000.0))
    noise = np.random.normal(0, 1, duration_samples)
    noise_power = np.mean(noise ** 2)
    target_noise_power = signal_power / (10 ** (snr_db / 10))
    noise = noise * np.sqrt(target_noise_power / noise_power)
    
    return noise

def repeat_or_trim_noise(noise, target_length):

    if len(noise) < target_length:
        repeat_times = target_length // len(noise) + 1
        noise = np.tile(noise, repeat_times)[:target_length]
    else:
        noise = noise[:target_length]
    return noise

def add_white_noise_beginning(input_file, output_file, snr_db=10):

    fs, audio = wavfile.read(input_file)
    audio = audio.astype(np.float32)
    audio_power = np.mean(audio ** 2)
    
    noise_duration_ms = len(audio) * 1000 / fs * 0.1  
    noise = generate_white_noise(noise_duration_ms, fs, snr_db, audio_power)
    
    audio = audio / np.max(np.abs(audio))
    modified_audio = np.concatenate([noise, audio])
    modified_audio = np.int16(modified_audio / np.max(np.abs(modified_audio)) * 32767)
    
    wavfile.write(output_file, fs, modified_audio)

def add_white_noise_middle(input_file, output_file, insertion_point_ms, snr_db=10):

    fs, audio = wavfile.read(input_file)
    audio = audio.astype(np.float32)
    audio_power = np.mean(audio ** 2)
    
    insertion_point_samples = int(fs * insertion_point_ms / 1000)
    
    first_part = audio[:insertion_point_samples]
    second_part = audio[insertion_point_samples:]
    
    noise_duration_ms = len(audio) * 1000 / fs * 0.1 
    noise = generate_white_noise(noise_duration_ms, fs, snr_db, audio_power)
    
    modified_audio = np.concatenate([first_part, noise, second_part])
    modified_audio = np.int16(modified_audio / np.max(np.abs(modified_audio)) * 32767)
    
    wavfile.write(output_file, fs, modified_audio)

def add_white_noise_throughout(input_file, output_file, snr_db=10):

    fs, audio = wavfile.read(input_file)
    audio = audio.astype(np.float32)
    audio_power = np.mean(audio ** 2)
    
    noise_duration_ms = len(audio) * 1000 / fs
    noise = generate_white_noise(noise_duration_ms, fs, snr_db, audio_power)

    modified_audio = audio + noise
    modified_audio = np.int16(modified_audio / np.max(np.abs(modified_audio)) * 32767)
    
    wavfile.write(output_file, fs, modified_audio)

def add_real_noise_throughout(input_file, output_file, noise_file, snr_db):

    noise_fs, noise_data = convert_to_mono(noise_file)

    fs_audio, audio = wavfile.read(input_file)
    audio = audio.astype(np.float32)
    audio_power = np.mean(audio ** 2)
    
    noise_power = np.mean(noise_data ** 2)
    target_noise_power = audio_power / (10 ** (snr_db / 10))
    noise_data = noise_data * np.sqrt(target_noise_power / noise_power)
    
    noise_data = repeat_or_trim_noise(noise_data, len(audio))
    
    modified_audio = audio + noise_data
    modified_audio = np.int16(modified_audio / np.max(np.abs(modified_audio)) * 32767)
    
    wavfile.write(output_file, fs_audio, modified_audio)

def cut_audio_middle(input_file, output_file, insertion_point_ms):

    fs, audio = wavfile.read(input_file)
    audio = audio.astype(np.float32)
    
    insertion_point_samples = int(fs * insertion_point_ms / 1000)
    
    cut_length = int(len(audio) * 0.1) 
    half_cut = cut_length // 2
    
    first_part = audio[:max(0, insertion_point_samples - half_cut)]
    second_part = audio[min(len(audio), insertion_point_samples + half_cut):]
    
    modified_audio = np.concatenate([first_part, second_part])
    modified_audio = np.int16(modified_audio / np.max(np.abs(modified_audio)) * 32767)
    
    wavfile.write(output_file, fs, modified_audio)

def calculate_insertion_point(row):

    try:
        input_file = row['segment_name']
        file_start, file_end = input_file.split('_')[1:3]
        file_end = file_end.strip('.wav')
        
        file_end_ms = int(file_end)

        cut_time_ms = float(row['cut_time'])
        
        audio_length_ms = file_end_ms - int(file_start)
        
        insertion_point_ms = audio_length_ms - (file_end_ms - cut_time_ms)
        return insertion_point_ms
    except ValueError:
        return np.nan

def process_audio_files(source_dir, csv_file_path, noise_file_path):

    variations = {
        'a': 'add_silence_beginning',
        'b': 'add_white_noise_beginning',
        'c': 'add_white_noise_middle',
        'd': 'add_white_noise_throughout',
        'e1': 'add_real_noise_high_snr',
        'e2': 'add_real_noise_low_snr',
        'f': 'cut_audio_middle'
    }
    
    for suffix, desc in variations.items():
        target_dir = f"{source_dir}_{suffix}"
        os.makedirs(target_dir, exist_ok=True)
        print(f"Created directory: {target_dir} for {desc}")
    
    try:
        df = pd.read_csv(csv_file_path)
        df['insertion_point'] = df.apply(calculate_insertion_point, axis=1)
        insertion_points = dict(zip(df['segment_name'], df['insertion_point']))
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        print("Proceeding without insertion points data. Middle insertion variations will be skipped.")
        insertion_points = {}
    
    for filename in os.listdir(source_dir):
        if not filename.endswith('.wav'):
            continue
        
        input_file = os.path.join(source_dir, filename)
        
        insertion_point_ms = insertion_points.get(filename, None)
        if insertion_point_ms is None:
            fs, audio = wavfile.read(input_file)
            insertion_point_ms = len(audio) * 1000 / (2 * fs)
        
        try:
            # Add silence to beginning
            output_file_a = os.path.join(f"{source_dir}_a", f"{filename[:-4]}_a.wav")
            add_silence_beginning(input_file, output_file_a)
            
            # Add white noise to beginning
            output_file_b = os.path.join(f"{source_dir}_b", f"{filename[:-4]}_b.wav")
            add_white_noise_beginning(input_file, output_file_b)
            
            # Add white noise in the middle
            output_file_c = os.path.join(f"{source_dir}_c", f"{filename[:-4]}_c.wav")
            add_white_noise_middle(input_file, output_file_c, insertion_point_ms)
            
            # Add white noise throughout
            output_file_d = os.path.join(f"{source_dir}_d", f"{filename[:-4]}_d.wav")
            add_white_noise_throughout(input_file, output_file_d)
            
            # Add real noise throughout (high SNR)
            output_file_e1 = os.path.join(f"{source_dir}_e1", f"{filename[:-4]}_e1.wav")
            add_real_noise_throughout(input_file, output_file_e1, noise_file_path, snr_db=15) 
            
            # Add real noise throughout (low SNR)
            output_file_e2 = os.path.join(f"{source_dir}_e2", f"{filename[:-4]}_e2.wav")
            add_real_noise_throughout(input_file, output_file_e2, noise_file_path, snr_db=5) 
            
            # Cut audio in the middle
            output_file_f = os.path.join(f"{source_dir}_f", f"{filename[:-4]}_f.wav")
            cut_audio_middle(input_file, output_file_f, insertion_point_ms)
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")

def main():

    source_dir = "experiment_samples"               
    csv_file_path = "audio_experiment_samples_cut.csv" 
    noise_file_path = "360703__eguobyte__large_crowd_medium_distance_stereo.wav"          
    
    process_audio_files(source_dir, csv_file_path, noise_file_path)

if __name__ == "__main__":
    main()