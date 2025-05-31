import openai
import os
import json
import time
import datetime
from dotenv import load_dotenv

def get_filepath_list(folderpath):
    pathname_list = []
    for path, subdirs, files in os.walk(folderpath):
        for name in files:
            path_name = os.path.join(path, name)
            if ".DS_Store" not in path_name:
                pathname_list.append(path_name)
    print("File collection done, in total there are ", len(pathname_list), " files")
    return pathname_list

def transcribe_files(pathname_list):
    openai.api_key = "api_key"
    transcription = []
    
    batch_size = 50
    for i in range(0, len(pathname_list), batch_size):
        batch_files = pathname_list[i:i+batch_size]
        
        batch_transcription = []
        for file in batch_files:
            with open(file, "rb") as audio_file:
                for j in range(5):
                    try:
                        transcript = openai.Audio.transcribe(mode="whisper-1", audio_file, language="en", timeout=120)
                        text = transcript["text"]
                        batch_transcription.append({"filename": file.split("/")[-1], "transcript": text})
                        break
                    except Exception as e:
                        print(f"Error occurred during transcription of {file}: {str(e)}")
                        time.sleep(10)
        
        transcription.extend(batch_transcription)
    return transcription

def main():
    APHASIA = input("Is this an aphasia group? (y/n) ")
    
    # Set up filenames for json files based on the group and date
    date = str(datetime.date.today())
    json_filename = date + ("_Aphasia_Whisper_transcript.json" if APHASIA == "y" else "_Control_Whisper_transcript.json")
    
    folderpath = input("Please enter the folder path: ")
    pathname_list = get_filepath_list(folderpath)

    transcriptions = transcribe_files(pathname_list)

    with open(json_filename, 'w') as outfile:
        json.dump(transcriptions, outfile)
    print("Transcripts have been successfully stored in json file...")

if __name__ == "__main__":
    main()
