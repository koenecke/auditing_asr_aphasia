# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import time
import concurrent.futures
import argparse

# [START speech_transcribe_batch_multiple_files_v2]
import json
import re
from typing import List
from google.cloud import storage
from google.api_core.client_options import ClientOptions
from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech
import os
import datetime
from dotenv import load_dotenv
from google.cloud import storage

def list_bucket_files(bucket_name):
    """List all file paths in a Google Cloud Storage bucket.

    Args:
        bucket_name (str): The name of the GCS bucket.

    Returns:
        List of file paths (str) in the bucket.
    """
    # Initialize the GCS client
    storage_client = storage.Client()

    # Get the bucket
    bucket = storage_client.bucket(bucket_name)

    file_paths = []

    # List all objects (files) in the bucket
    blobs = bucket.list_blobs()

    for blob in blobs:
        # The file path is the blob's name
        file_paths.append(f"gs://{bucket_name}/{blob.name}")


    return file_paths



def transcribe_batch_multiple_files_v2(
    project_id: str,
    gcs_uris: List[str],
    gcs_output_path: str,
    model_name: str,
    json_filename: str,
) -> cloud_speech.BatchRecognizeResponse:
    """Transcribes audio from a Google Cloud Storage URI.

    Args:
        project_id: The Google Cloud project ID.
        gcs_uris: The Google Cloud Storage URIs to transcribe.
        gcs_output_path: The Cloud Storage URI to which to write the transcript.

    Returns:
        The BatchRecognizeResponse message.
    """
    # Instantiates a client
    client = SpeechClient(
        client_options=ClientOptions(
            api_endpoint="us-central1-speech.googleapis.com",
        )
    )

    config = cloud_speech.RecognitionConfig(
        auto_decoding_config=cloud_speech.AutoDetectDecodingConfig(),
        language_codes=["en-US"],
        model=model_name, # this is the model for Google Chirp
    )

    files = [
        cloud_speech.BatchRecognizeFileMetadata(uri=uri)
        for uri in gcs_uris
    ]

    request = cloud_speech.BatchRecognizeRequest(
        recognizer=f"projects/{project_id}/locations/us-central1/recognizers/_",
        config=config,
        files=files,
        recognition_output_config=cloud_speech.RecognitionOutputConfig(
            gcs_output_config=cloud_speech.GcsOutputConfig(
                uri=gcs_output_path,
            ),
        ),
    )

    # Transcribes the audio into text
    operation = client.batch_recognize(request=request)

    print("Waiting for operation to complete...")
    # Define the maximum number of retries
    max_retries = 3
    retry_delay_seconds = 10  # Delay between retries in seconds

    for retry in range(max_retries):
        try:
            print(f"Attempt {retry + 1}: Waiting for operation to complete...")
            response = operation.result(timeout=300)
            break  # Operation completed successfully, exit the retry loop
        except concurrent.futures.TimeoutError:
            print(f"Attempt {retry + 1}: Operation timed out. Retrying in {retry_delay_seconds} seconds...")
            time.sleep(retry_delay_seconds)
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            break  # Handle other exceptions and exit the retry loop

    if retry == max_retries - 1:
        print("Maximum number of retries reached. Operation did not complete successfully.")
    else:
        print("Operation finished successfully.")
    

    print("Operation finished. Fetching results from:")

    transcript_object_list = []
    failed_job_list = []

    for uri in gcs_uris:
        file_results = response.results[uri]
        # raise key error if the file is not transcribed successfully
        if file_results.error:
            response = operation.result(timeout=300)
     
        output_bucket, output_object = re.match(
            r"gs://([^/]+)/(.*)", file_results.uri
        ).group(1, 2)
        # print(output_bucket, output_object)

        storage_client = storage.Client()

        # Fetch results from Cloud Storage
        bucket = storage_client.bucket(output_bucket)
        blob = bucket.blob(output_object)
        results_bytes = blob.download_as_bytes()
        batch_recognize_results = cloud_speech.BatchRecognizeResults.from_json(
            results_bytes, ignore_unknown_fields=True
        )
        filename = uri.split("/")[-1]
        transcript = ""
    
        for result in batch_recognize_results.results:
            if result.alternatives:
                # print(f" Transcript: {result.alternatives[0].transcript}")
                transcript += result.alternatives[0].transcript
            else:
                print("No transcription alternatives found.")
                transcript +=""

     
        object={"filename":filename, "transcript":transcript}

        transcript_object_list.append(object)
 
    print("Saving transcripts to json file...")
    with open(json_filename, 'w') as f:
        json.dump(transcript_object_list, f)

    if len(failed_job_list) > 0:
        failed_job_json_filename = "Google_"+model_name+"_failed_job_list.json"
        # check if there is already a json file for failed jobs
        if not os.path.exists(failed_job_json_filename):
            with open(failed_job_json_filename, 'w') as outfile:
                json.dump(failed_job_list, outfile)
                print(f"there are {len(failed_job_list)} failed jobs, please check the json file for details")
        else:
            # open the json file and read the data first
            with open(failed_job_json_filename) as json_file:
                data = json.load(json_file)
                if data is None:
                    data = []

                # Append failed_job_list to data
                data.append(failed_job_list)

                # Now new_failed_job_list should contain the updated data
                new_failed_job_list = data

            # append the new failed jobs to the list
                new_failed_job_list = data.append(failed_job_list)
            # write the updated list to the json file
            with open(failed_job_json_filename, 'w') as outfile:
                json.dump(new_failed_job_list, outfile)
                print(f"there are {len(failed_job_list)} failed jobs, please check the json file for details")
    else:
        print("All jobs have been transcribed successfully...")

    return response



def transcribe_v2_chirp(
    project_id: str,
    audio_file: str,
) -> cloud_speech.RecognizeResponse:
    """Transcribe a single local audio file with Google Chirp."""
    # Instantiates a client
    client = SpeechClient(
        client_options=ClientOptions(
            api_endpoint="us-central1-speech.googleapis.com",
        )
    )

    # Reads a file as bytes
    with open(audio_file, "rb") as f:
        content = f.read()

    config = cloud_speech.RecognitionConfig(
        auto_decoding_config=cloud_speech.AutoDetectDecodingConfig(),
        language_codes=["en-US"],
        model="chirp",
    )

    request = cloud_speech.RecognizeRequest(
        recognizer=f"projects/{project_id}/locations/us-central1/recognizers/_",
        config=config,
        content=content,
    )
    # Transcribes the audio into text
    response = client.recognize(request=request)
    transcript = ""
    for result in response.results:
        transcript += result.alternatives[0].transcript
    print(transcript)
    return transcript

def transcribe_multiple_files(project_id,folder_path):
    '''
    This function transcribes all the files in a local folder
    '''
    # get all the files in the folder
    files = os.listdir(folder_path)
    # get the full path for each file
    audio_path_names = [folder_path+file for file in files]
    for audio_file in audio_path_names:
        transcribe_v2_chirp(project_id, audio_file)


def main():
    APHASIA = input("Is this an aphasia group? (y/n) ")
    if APHASIA == "y":
        APHASIA = True
    else:
        APHASIA = False

    # set up filenames for json files to store job ids and transcripts
    date = str(datetime.date.today())
    if APHASIA:
        json_file_for_transcript = date+"_Aphasia_GoogleChirp_transcript"
    else:
        json_file_for_transcript = date+"_Control_GoogleChirp_transcript"

    # Get the current directory path
    current_directory = os.getcwd()
    print("Current directory:", current_directory)

    # Create a data folder in this directory
    data_folder_path = os.path.join(current_directory, "data/google_transcripts")
    os.makedirs(data_folder_path, exist_ok=True)  # Create the folder if it doesn't exist


    ## set up google credentials
    ## Step 1: Initialize the Google Cloud SDK by running the ‘gcloud init’. This will prompt you to log in to your Google Cloud account and set up the default credentials for your project.
    ## Step 2: Activate the appropriate service account by running following command ‘gcloud auth activate-service-account --key-file=YOUR_SERVICE_ACCOUNT_KEY.json(Replace YOUR_SERVICE_ACCOUNT_KEY.json with the path to your service account key file.).
    ## Step 3: Verify your authentication by running this command ‘gcloud auth list’. This should display the active credentials you’ve set up.
    ## Step 4: Run your python script.
    
    load_dotenv()
    credential_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    project_id = os.getenv("GOOGLE_PROJECT_ID")
    print(credential_path)
    print(project_id)
    gsc_uris= list_bucket_files("aphasia_segment") # change bucket name 

    gsc_output_path = "gs://aphasia_transcript" # change to new bucket name that stores transcripts

    # break gsc_uris into batches and each batch size is 15
    total_files = len(gsc_uris) 
    desired_batch_size = 15 # set batch size to 15 because this is the maximum number of files that can be transcribed in one batch

    # Calculate the batch size based on the total number of files
    batch_size = min(total_files, desired_batch_size)


    # Split the list of files into batches
    batch_number = 0  
    for batch_index in range(0, total_files + 1, batch_size):
        batch_number += 1
        if batch_number<501:
            continue
        print("------Starting batch number ", batch_number)
        if batch_index == total_files:
            print("This is the end of the list")
            break
        batch_pathnames_list = gsc_uris[batch_index: min(batch_index + batch_size, total_files + 1)]

        file = json_file_for_transcript+"_Batch_"+ str(batch_number) + ".json"
        json_filename = os.path.join(data_folder_path, file)

        transcribe_batch_multiple_files_v2(project_id, batch_pathnames_list, gsc_output_path,"chirp",json_filename)


if __name__ == "__main__":
    main()

