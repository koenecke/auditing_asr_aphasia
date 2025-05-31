'''
File to transcribe audio files using assemblyAI

Notes: Assembly AI has number of files limit per submission. 
Thus this code submits files by batches. 
'''

import requests
import os
import json
import pandas as pd
import utils
import time
import numpy as np
import datetime

def get_files_paths (folder_path):
    pathname_list = []
    for path, subdirs, files in os.walk(folder_path):
        
        for name in files:
            path_name = os.path.join(path, name)
            if ".DS_Store" not in path_name:
                pathname_list.append(path_name)

    for path, subdirs, files in os.walk(folder_path):
            
            for name in files:
                path_name = os.path.join(path, name)
                if ".DS_Store" not in path_name:
                    pathname_list.append(path_name)
    print("File collection done")
    return pathname_list

#####NEEDS TO CHANGE FOR EVERY EXPERIMENT####
#every time just limit the max to 50 files because assembly AI doesn't process too many files at once
# pathname_list = pathname_list[0:50] 


def read_file(filename, chunk_size=5242880):
    with open(filename, 'rb') as _file:
        while True:
            data = _file.read(chunk_size)
            if not data:
                break
            yield data

#function to transcribe individual file in Assembly ai
def assembly_ai_transcribe (upload_url,headers):
    transcript_response = utils.request_transcript(upload_url, headers)
    print(f"Calling assembly ai for {upload_url}")

    if transcript_response['status']=="error":
        print(f"Error: {transcript_response['status']}")
        return "N/A"
    
    timeout = 60 * 2 # 2 minutes

    # Create a polling endpoint that will let us check when the transcription is complete
    polling_endpoint = utils.make_polling_endpoint(transcript_response)

    # Wait until the transcription is complete
    try:
        utils.wait_for_completion(polling_endpoint, headers, timeout, time.time())
    except Exception as e:
        if (e is TimeoutError):
            print("Timed out waiting for completion")
        return "N/A"
    except json.decoder.JSONDecodeError:
        print("JSON decoding error")
        return "N/A"

    # Request the paragraphs of the transcript
    paragraphs = utils.get_paragraphs(polling_endpoint, headers)

    # Save and print transcript
    #with open('transcript.txt', 'w') as f:
    current_transcript = ""
    for para in paragraphs:
        #print(para['text'])
        current_transcript += para['text']
        #f.write(para['text'] + '\n')
    # print(current_transcript)
    return(current_transcript)



def submit_file(filename,header):
    print(f"      Submitting {filename}")
    response = requests.post('https://api.assemblyai.com/v2/upload',
                        headers=header,
                        data=read_file(filename))
    return response

def get_file_transcript(response,header):    
    transcript = assembly_ai_transcribe(response.json(),headers= header)
    return transcript

def append_transcripts_to_json(transcripts, filenames, filename):
    '''Append a list of transcripts to an array json file.'''
    with open(filename, 'r') as f:
        current_json = json.load(f)
    objs = [{"filename": filenames[i], "transcript": transcripts[i]} for i in range(len(transcripts))]
    
    if current_json['records'] is None:
        current_json['records'] = []

    current_json['records'].extend(objs)
    with open(filename, 'w') as f:
        json.dump(current_json, f)
    f.close()


def batch_transcribe(pathnames,header):
    '''Transcribe a batch and wait for completion. Return a list of transcripts for this batch.'''

    # Submit the files
    responses = [submit_file(filename,header) for filename in pathnames]

    # Get the transcripts
    transcripts = [get_file_transcript(response,header) for response in responses]

    return transcripts

def write_batch_to_excel(filenames, batch_transcripts):
    '''
    Write a batch of transcripts to an excel file.
    '''

    with pd.ExcelWriter("assembly-transcriptions-aphasia-3-26-23.xlsx", mode="a", engine="openpyxl") as writer:
        batch_df = pd.DataFrame()
        batch_df['filenames']=  filenames
        batch_df['transcripts']=  batch_transcripts
        batch_df.to_excel(writer, sheet_name='Sheet1', index=False)
    writer.close()


def get_filenames(pathname_list):

    '''
    Get the filenames from a list of pathnames.
    '''
    filenames = []
    for pathname in pathname_list:
        filenames.append(pathname.split(sep = "/")[-1])
    return filenames


def get_json_file(filename, mode="r"):
    '''Get the json file as a dictionary.'''
    with open(filename, mode) as json_file:
        try:
            return json.load(json_file)
        except json.decoder.JSONDecodeError:
            return {}

def get_processed_files(JSON_filename):

    '''
    Get the list of files that have already been processed.
    @para JSON_filename: the name of the json file containing the processed files
    '''
    processed_filedata = get_json_file(JSON_filename)
    processed_files = []
    for file in processed_filedata:
        processed_files.append(file.get("filename"))
    return processed_files

def get_processed_files_inExcel(filename):
    '''Get the list of files that have already been processed in Excel'''
    df = pd.read_csv(filename)
    processed_files = df['Filename'].tolist()
    return processed_files

def get_assemblyai_transcripts(index,JSON_file,new_JSON_file,header):
    '''Get the transcripts from assemblyai and write them to an excel file.
    index: the index of the job list to process
    JSON_file: the json file containing the processed files
    '''
    filenames_list = get_filenames(pathname_list)
    filenames_lists = np.array_split(filenames_list, 10)
    job_lists = np.array_split(pathname_list, 10)
    
    if os.path.exists(JSON_file):
        existing_files = get_processed_files(JSON_file)
    else:
        existing_files = []

    count = 0
    results = []
    for ind,job_id in enumerate(job_lists[index]):
        
        file = filenames_lists[index][ind]
        filename = file.split(sep = "/")[-1]
        if filename in existing_files:
            print(filename,"this file has been processed")
            continue
        count +=1
        print("this is file number ",count, file)
        response = requests.post('https://api.assemblyai.com/v2/upload',
                            headers=header,
                            data=read_file(job_id))
        transcript = assembly_ai_transcribe(response.json(),headers= header)
        object  = {"filename": file, 
                "transcript": transcript}
        results.append(object)
        with open(new_JSON_file, "w") as p:
            p.write(json.dumps(results))



def get_assemblyai_transcripts_fast(path_names,JSON_file,header):
    '''Get the transcripts from assemblyai and write them to json file
    # there is no check of existing files
    '''
    total = len(path_names)


    # check if the json file exists
    if os.path.exists(JSON_file):
        results = get_json_file(JSON_file)

    # if not, create a new json file
    else:
        results = []

    for ind,job_id in enumerate(path_names):
       
        file = path_names[ind]
        filename = file.split(sep = "/")[-1]

        print("this is file number ",ind+1, filename, "out of ", total)
        response = requests.post('https://api.assemblyai.com/v2/upload',
                            headers=header,
                            data=read_file(job_id))
        transcript = assembly_ai_transcribe(response.json(),headers= header)
        object  = {"filename": filename, 
                "transcript": transcript}
       
        results.append(object)
        
        # if the number of files processed is a multiple of 20, write to json file
        if ind % 20 == 0:
            print("updating json file")
            with open (JSON_file, "w") as p:
                p.write(json.dumps(results))

    with open(JSON_file, "w") as p:
        p.write(json.dumps(results))

def main():
    '''Main function to run the assemblyai transcriptions'''

    APHASIA = input("Is this an aphasia group? (y/n) ")
    if APHASIA == "y":
        APHASIA = True
    else:
        APHASIA = False

    # set up filenames for json files to store job ids and transcripts
    date = str(datetime.date.today())
    if APHASIA:
       filename_prefix = date+"_Aphasia_AssemblyAI_transcript_"
    else:
        filename_prefix = date+"_Control_AssemblyAI_transcript_"
    # folder_path = input("Enter the path to the folder containing the audio files: ")
    # JSON_file = input("Enter the name of the json file containing the processed files: ")


    # set up credentials 
    HEADER = {'authorization': "key"} 
    

    # get file paths 
    FOLDER_PATH = input("Enter the path to the folder containing the audio files: ")
    file_paths = get_files_paths(FOLDER_PATH)


    total_files = len(file_paths) 
    desired_batch_size = 32

    # Calculate the batch size based on the total number of files
    batch_size = min(total_files, desired_batch_size)
    print("Batch size: ", batch_size)
    # Split the list of files into batches
    batch_number = 0  
    for batch_index in range(1, total_files + 1, batch_size):
        batch_number += 1
        if batch_number < 173:
            print("---Skipping batch number ", batch_number)
            continue
        print("------Starting batch number ", batch_number)
        batch_pathnames_list = file_paths[batch_index: min(batch_index + batch_size, total_files + 1)]
        
        json_filename = filename_prefix + str(batch_index) + ".json"
        get_assemblyai_transcripts_fast(batch_pathnames_list, json_filename, HEADER)

        if batch_index + batch_size >= total_files:
            break  # Stop processing if it's the last batch

if __name__ == "__main__":
    main()
      