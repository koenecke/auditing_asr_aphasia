import boto3
import json
import time
import os
import requests
import datetime

def transcribe_file(job_name, file_uri):
    transcribe = boto3.client('transcribe')
    transcribe.start_transcription_job(
        TranscriptionJobName=job_name,
        Media={'MediaFileUri': file_uri},
        MediaFormat='wav',
        LanguageCode='en-US'
    )

    while True:
        status = transcribe.get_transcription_job(TranscriptionJobName=job_name)
        if status['TranscriptionJob']['TranscriptionJobStatus'] in ['COMPLETED', 'FAILED']:
            break
        time.sleep(15)

    if status['TranscriptionJob']['TranscriptionJobStatus'] == 'COMPLETED':
        transcript_uri = status['TranscriptionJob']['Transcript']['TranscriptFileUri']
        response = requests.get(transcript_uri)
        data = response.json()
        return data['results']['transcripts'][0]['transcript']
    else:
        return None

def process_file(file_name):
    job_name = f"{file_name}-{time.time()}"
    file_uri = f"s3://nyuaphasia/{file_name}"
    transcription = transcribe_file(job_name, file_uri)
    return transcription

def upload_files_to_s3(directory_path, bucket_name):
    s3_client = boto3.client('s3')
    uploaded_files = []

    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.wav'):
                file_path = os.path.join(root, file)
                s3_client.upload_file(file_path, bucket_name, file)
                uploaded_files.append(file)

    return uploaded_files

def main():
    # Setting up your AWS credentials
    os.environ['AWS_ACCESS_KEY_ID'] = 'access_key_id'
    os.environ['AWS_SECRET_ACCESS_KEY'] = 'secret_access_key'
    
    APHASIA = input("Is this an aphasia group? (y/n) ")
    if APHASIA == "y":
        APHASIA = True
    else:
        APHASIA = False

    date = str(datetime.date.today())
    if APHASIA:
        json_file_for_transcript = date+"_Aphasia_AWS_transcript.json"
    else:
        json_file_for_transcript = date+"_Control_AWS_transcript.json"

    directory_path = input("Please enter the directory path: ")
    bucket_name = 'nyuaphasia'

    uploaded_files = upload_files_to_s3(directory_path, bucket_name)
    transcription_results = []

    failed_files_list = []

    for file_name in uploaded_files:
        transcription = transcribe_file(file_name, f"s3://{bucket_name}/{file_name}")

        if transcription:
            transcription_results.append({
                'segment_name': file_name,
                'aws_transcription': transcription
            })
        else:
            failed_files_list.append(file_name)

    if failed_files_list:
        print(f"There are {len(failed_files_list)} failed files. Please check them individually.")
    else:
        print("All files have been transcribed successfully.")

    with open(json_file_for_transcript, 'w') as json_file:
        json.dump(transcription_results, json_file)
    print("Transcription results have been successfully stored in JSON file.")

if __name__ == '__main__':
    main()