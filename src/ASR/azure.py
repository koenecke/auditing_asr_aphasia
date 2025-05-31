import os
import json
import azure.cognitiveservices.speech as speechsdk
import datetime

def get_filepath_list(folderpath):
    pathname_list = []
    for path, subdirs, files in os.walk(folderpath):
        for name in files:
            path_name = os.path.join(path, name)
            if ".DS_Store" not in path_name and name.endswith(".wav"):
                pathname_list.append(path_name)
    return pathname_list

def transcribe_files_with_azure(pathname_list, speech_config):
    transcriptions = []

    for audio_file_path in pathname_list:
        filename = audio_file_path.split("/")[-1]
        audio_input = speechsdk.audio.AudioConfig(filename=audio_file_path)
        speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_input)

        result = speech_recognizer.recognize_once()
        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            transcriptions.append({"segment_name": filename, "azure_transcription": result.text})
        elif result.reason == speechsdk.ResultReason.NoMatch:
            transcriptions.append({"segment_name": filename, "azure_transcription": "No speech could be recognized"})
        elif result.reason == speechsdk.ResultReason.Canceled:
            cancellation_details = result.cancellation_details
            msg = "Speech Recognition canceled: {}".format(cancellation_details.reason)
            if cancellation_details.reason == speechsdk.CancellationReason.Error:
                msg += ", Error details: {}".format(cancellation_details.error_details)
            transcriptions.append({"segment_name": filename, "azure_transcription": msg})
    
    return transcriptions

def main():
    APHASIA = input("Is this an aphasia group? (y/n) ")
    date = str(datetime.date.today())
    json_filename = date + ("_Aphasia_Azure_transcript.json" if APHASIA == "y" else "_Control_Azure_transcript.json")
    
    speech_key, service_region = "key", "region"
    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
    
    folderpath = input("Please enter the folder path: ")
    pathname_list = get_filepath_list(folderpath)

    transcriptions = transcribe_files_with_azure(pathname_list, speech_config)

    with open(json_filename, 'w') as outfile:
        json.dump(transcriptions, outfile)
    print("Transcripts have been successfully stored in json file...")

if __name__ == "__main__":
    main()
