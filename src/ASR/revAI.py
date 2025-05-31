from requests import HTTPError
import os
import json
import pandas as pd
import xlsxwriter as writer
import utils
from rev_ai import apiclient
import re
import logging
from time import sleep
import datetime
def get_filepath_list(folderpath):
    '''Input folder path and returns a list of file paths'''
    pathname_list = []
    for path, subdirs, files in os.walk(folderpath):
        for name in files:
            path_name = os.path.join(path, name)
            if ".DS_Store" not in path_name:
                pathname_list.append(path_name)
    print("File collection done, in total there are ",
          len(pathname_list), " files")

    return pathname_list


def submit_job(pathname_list, client, json_file_for_job_id):

    last_uploaded_index = 0

    filename_ids = []  # used to store filename and their job ids object
    job_id_list = []  # used to store job ids

    for pathname in pathname_list:
        filePath = pathname
        filename = pathname.split("/")[-1]
        last_uploaded_index += 1

        print(f"Submitting {filePath}, number {last_uploaded_index}")
        # send a local file
        job = client.submit_job_local_file(filePath)
        job_id_list.append(job.id)
        object = {"filename": filename,
                  "job_id": job.id,
                  "apiVersion": "RevAI"}
        filename_ids.append(object)

    # open a new file and store the job ids
    with open(json_file_for_job_id, 'w') as outfile:
        json.dump(filename_ids, outfile)
    print("Job ids have been sucessfully stored in json file...")


def main():

    APHASIA = input("Is this an aphasia group? (y/n) ")
    if APHASIA == "y":
        APHASIA = True
    else:
        APHASIA = False

    # set up filenames for json files to store job ids and transcripts
    date = str(datetime.date.today())
    if APHASIA:
        json_file_for_job_id = "Aphasia_RevAI_job_id.json"
        json_file_for_transcript = date+"_Aphasia_RevAI_transcript.json"
        failed_job_file = "Aphasia_RevAI_failed_job_list.json"
    else:
        json_file_for_job_id = "Control_RevAI_job_id.json"
        json_file_for_transcript = date+"_Control_RevAI_transcript.json"
        failed_job_file = "Control_RevAI_failed_job_list.json"

    # acquire token for RevAI access from environment variable

    token = "token"
    # create your client
    client = apiclient.RevAiAPIClient(token)

    # # get list of file paths
    folderpath = input("Please enter the folder path: ")
    pathname_list = get_filepath_list(folderpath)

    # submit jobs and store job ids
    submit_job(pathname_list, client, json_file_for_job_id)

    # identify path for saved json
    current_directory = os.getcwd()
    json_file_for_job_id = os.path.join(current_directory, json_file_for_job_id)

    # check if the path is correct
    print("Checking if the path to the json file is correct...")
    if not os.path.exists(json_file_for_job_id):
        print("The path to the json file is incorrect, please check")
        return
    
    # if the path is correct, get the job ids
    with open(json_file_for_job_id) as f:
        data = json.load(f)
    job_id_list = [job["job_id"] for job in data]  
    job_filenames = [job["filename"] for job in data]

    # check job status
    print("Checking job status...")
    failed_job_list = []
    
    # check the last 300 jobs
    for index,job_id in enumerate(job_id_list[-300:]):
        print(f"Checking job number", index+1, "out of", len(job_id_list[-300:]))
        untranscribed = True
        while untranscribed:
            try:
                job_details = client.get_job_details(job_id)
                if job_details.status.name == "TRANSCRIBED":
                    untranscribed = False
                    break
                if job_details.status.name == "FAILED":
                    print(job_details.failure)
                    failed_job_list.append(job_id)
                    break
                if job_details.status.name == "IN_PROGRESS":
                    print("Job is still in progress")
                    continue
                sleep(30)
            except HTTPError:
                failed_job_list.append(job_id)
                break
    print("Job status check done...")
    # if there are failed jobs, write them to a file
    if len(failed_job_list) > 0:
        with open(failed_job_file, 'w') as outfile:
            json.dump(failed_job_list, outfile)
        print(f"there are {len(failed_job_list)} failed jobs, please check the json file for details")
    else:
        print("All jobs have been transcribed successfully...")
   
    #get transcripts
    job_transcripts = []
    for index,job_id in enumerate(job_id_list):
        print(f"Getting transcript number", index+1, "out of", len(job_id_list))
        job_details = client.get_job_details(job_id)
        # Get transcript as text
        if job_details.status.name == "TRANSCRIBED":
            transcript_text = client.get_transcript_text(job_id)
        else:
            transcript_text = "FAILED (n/a)"
        object = {"job_id": job_id,
                "filename": job_filenames[index],
                "transcript": transcript_text}
        job_transcripts.append(object)

    with open(json_file_for_transcript, 'w') as outfile:
        json.dump(job_transcripts, outfile)
    print("Transcripts have been sucessfully stored in json file...")

if __name__ == "__main__":
    main()