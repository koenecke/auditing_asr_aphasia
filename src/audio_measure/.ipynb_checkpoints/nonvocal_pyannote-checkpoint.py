import os
import csv
from pyannote.audio import Pipeline
from pydub import AudioSegment

vad_pipeline = Pipeline.from_pretrained("pyannote/voice-activity-detection", use_auth_token="token")

def get_nonvocal_duration(file_path):
    """Calculate the total nonvocal duration of an audio file."""
    audio = AudioSegment.from_file(file_path)
    total_duration_ms = len(audio)
    vad = vad_pipeline({"audio": file_path})
    vocal_duration_ms = sum(segment.duration for segment in vad.get_timeline())
    nonvocal_duration_ms = total_duration_ms - vocal_duration_ms
    return nonvocal_duration_ms

def process_directory(directory, group):
    """Process all files in a directory and return a list of results."""
    results = []
    for filename in os.listdir(directory):
        if filename.endswith(".wav"): 
            file_path = os.path.join(directory, filename)
            nonvocal_duration = get_nonvocal_duration(file_path)
            results.append([group, filename, nonvocal_duration])
    return results

aphasia_results = process_directory("Aphasia_segment/", "aphasia")
control_results = process_directory("Control_segment/", "control")
all_results = aphasia_results + control_results

with open("nonvocal_duration_pyannote.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["group", "filename", "nonvocal_duration"])
    writer.writerows(all_results)

print("Processing complete. Results saved to 'nonvocal_duration_results.csv'.")
