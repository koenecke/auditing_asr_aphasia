import argparse
import pandas as pd
import numpy as np
import re
import json
import datetime
import os


def detect_diarization_pattern(input_string):

    pattern = r"Speaker \d"
    
    if not isinstance(input_string, str) or pd.isna(input_string):
        return 0
    
    matches = re.findall(pattern, input_string)
    
    matches = list(set(matches))
    return len(matches)

def detect_interviewer_speech(input_string):

    patterns = [
        "tell me",
        "tell me a story",
        "take a look at",
        "how to make a peanut butter sandwich"
    ]
    
    if isinstance(input_string, str) and not pd.isna(input_string):
        input_string = input_string.lower()
        for pattern in patterns:
            matches = re.findall(pattern, input_string)
            if len(matches) > 0:
                return True
    
    return False

def calculate_word_count(input_string):

    if pd.isna(input_string):
        return None
    
    if isinstance(input_string, str) and not pd.isna(input_string):
        input_string = input_string.lower()
        word_count = len(input_string.split())
        return word_count
    else:
        return 0

def detect_non_english_word(input_string):

    pattern = r"[^a-zA-Z0-9\s]"
    
    if isinstance(input_string, str) and not pd.isna(input_string):
        input_string = input_string.lower()
        matches = re.findall(pattern, input_string)
        
        matches = list(set(matches))
        return len(matches) > 0
    
    return False

def process_asr_data(asr_data):

    asr_list = [
        "RevAI", "AWS", "Azure", "GoogleChirp", "GoogleTelephony", 
        "AssemblyAI", "GoogleLong", "Whisper"
    ]
    
    asr_wordcount_dict = []
    
    for _, row in asr_data.iterrows():
        obj = {}
        obj["filename"] = row["filename"]
        obj["Group"] = row["Group"]
        obj["groundtruth"] = row["groundtruth_RF"]
        gt_word_count = calculate_word_count(row["groundtruth_RF"])
        obj["gt_word_count"] = gt_word_count
        
        string_detect_check = []
        
        for service_name in asr_list:
            obj[service_name+"_RF"] = row[service_name+"_RF"]
            check_hallucination = detect_non_english_word(row[service_name+"_RF"])
            obj[service_name+"_sus_hallucination"] = check_hallucination
            service_word_count = calculate_word_count(row[service_name+"_RF"])
            
            if service_word_count is not None:
                obj[service_name+"_word_count"] = service_word_count
                
                word_count_diff = service_word_count - gt_word_count
                obj[service_name+"_word_count_diff_from_gt"] = word_count_diff
                
                duration = row["duration"]/1000
                gt_word_per_second = gt_word_count / duration
                obj["duration"] = duration
                obj["gt_word_per_second"] = round(gt_word_per_second, 2)
                
            check = detect_interviewer_speech(row[service_name+"_V1"])
            string_detect_check.append(check)
        
        if string_detect_check.count(True) >= 3:
            obj["string_detect_check"] = True
        elif string_detect_check.count(None) == 1:
            obj["string_detect_check"] = "NA"
        else:
            obj["string_detect_check"] = False
            
        asr_wordcount_dict.append(obj)
    
    word_count_df = pd.DataFrame(asr_wordcount_dict)
    
    min_count_diff = 5
    more_than_three_asr_word_count_higher_than_gt = []
    
    for _, row in word_count_df.iterrows():
        count = 0
        for service_name in asr_list:
            col_name = f"{service_name}_word_count_diff_from_gt"
            if col_name in row and row[col_name] > min_count_diff:
                count += 1
        
        more_than_three_asr_word_count_higher_than_gt.append(count >= 4)
    
    word_count_df["more_than_three_asr_word_count_higher_than_gt"] = more_than_three_asr_word_count_higher_than_gt
    
    gt_word_per_second_df = word_count_df['gt_word_per_second']
    
    word_count_df["speech_speed_check"] = ~gt_word_per_second_df.between(
        gt_word_per_second_df.quantile(.05), 
        gt_word_per_second_df.quantile(.95)
    )
    
    return word_count_df

def process_azure_diarization(azure_aphasia_file, azure_control_file):

    with open(azure_aphasia_file, "r") as infile:
        data1 = json.load(infile)
        
    with open(azure_control_file, "r") as infile:
        data2 = json.load(infile)
        
    data = {**data1, **data2}
    
    objects = []
    for key in data.keys():
        speaker_number = []
        for item in data[key]:
            speaker_number.append(item["Speaker_ID"])
        speaker_number = len(list(set(speaker_number)))
        filename = key.split(".")[0] + ".wav"
        obj = {"filename": filename, "azure_speaker_number": speaker_number}
        objects.append(obj)
        
    azure_speaker_number_df = pd.DataFrame(objects)
    
    azure_speaker_number_df["azure_diarization_check"] = np.where(
        azure_speaker_number_df["azure_speaker_number"] > 1, 
        True, 
        np.where(azure_speaker_number_df["azure_speaker_number"] == 1, False, "NA")
    )
    
    return azure_speaker_number_df

def process_revai_diarization(joined_transcripts_file):

    df = pd.read_csv(joined_transcripts_file)
    number_of_matches = []
    
    for _, row in df.iterrows():
        if not pd.isna(row["RevAI"]) and isinstance(row["RevAI"], str):
            num_matches = detect_diarization_pattern(row["RevAI"])
            number_of_matches.append(num_matches)
        else:
            number_of_matches.append(pd.NA)
            
    df["num_matches"] = number_of_matches
    df["RevAI_diarization_check"] = np.where(
        df["num_matches"] > 1, 
        True, 
        np.where(df["num_matches"] == 1, False, "NA")
    )
    
    rev_AI_check_df = df[["filename", "Group", "RevAI_diarization_check"]]
    return rev_AI_check_df

def create_summary_stats(final_df, interviewer_check_df):

    print("\nSummary of check metrics:")
    print("=" * 50)

    print(f"String detect check: {final_df['string_detect_check'].value_counts().to_dict()}")
    
    print(f"Speech speed check: {final_df['speech_speed_check'].value_counts().to_dict()}")

    print(f"Word count difference check: {final_df['more_than_three_asr_word_count_higher_than_gt'].value_counts().to_dict()}")

    print(f"RevAI diarization check: {final_df['RevAI_diarization_check'].value_counts().to_dict()}")
    
    print(f"Azure diarization check: {final_df['azure_diarization_check'].value_counts().to_dict()}")
    
    if 'manual_check' in interviewer_check_df.columns:
        print(f"Manual check: {interviewer_check_df['manual_check'].value_counts().to_dict()}")
    
    if 'no_tag' in interviewer_check_df.columns:
        print(f"No tag check: {interviewer_check_df['no_tag'].value_counts().to_dict()}")


def filter_files(final_df, interviewer_check_df):

    print("\nFiltering files based on check metrics:")
    print("=" * 50)
    
    final_df['Interviewer_Speech_Included'] = False
    
    one_of_true = ["True", "true", True]
    filter_conditions = [
        ("manual_check", one_of_true),
        ("string_detect_check", one_of_true),
        ("RevAI_diarization_check", one_of_true),
        ("azure_diarization_check", one_of_true),
        ("more_than_three_asr_word_count_higher_than_gt", one_of_true),
        ("no_tag", one_of_true)
    ]
    
    filtered_df = final_df.copy()
    
    for column, value in filter_conditions:
        if column in filtered_df.columns:
            before_count = filtered_df.shape[0]
            filtered_df = filtered_df[(~filtered_df[column].isin(value)) & 
                                   ~(filtered_df.get("manual_check", pd.Series([False] * len(filtered_df))).isin([
                                       "False", "false", False
                                   ]))]
            after_count = filtered_df.shape[0]
            print(f"After filtering by {column}, removed {before_count - after_count} files, {after_count} files left")
    
    print("\nAfter all filtering:")
    if "Group" in filtered_df.columns:
        print(filtered_df["Group"].value_counts().to_dict())
        
    return filtered_df


def main():
    parser = argparse.ArgumentParser(description='Detect interviewer speech in audio transcripts.')
    parser.add_argument('--input', required=True, help='Input CSV file with ASR data')
    parser.add_argument('--output', required=True, help='Output CSV file path')
    parser.add_argument('--azure_dia', required=True, nargs=2, 
                        help='Azure diarization JSON files (aphasia and control)')
    parser.add_argument('--joined', required=True, help='Joined transcripts CSV file')
    parser.add_argument('--manual_check', required=False, help='Manual check CSV file (optional)')
    
    args = parser.parse_args()
    
    print(f"Processing input file: {args.input}")
    
    asr_data = pd.read_csv(args.input)
    
    word_count_df = process_asr_data(asr_data)
    
    azure_speaker_number_df = process_azure_diarization(args.azure_dia[0], args.azure_dia[1])
    
    rev_AI_check_df = process_revai_diarization(args.joined)
    
    revAI_check = rev_AI_check_df[["filename", "RevAI_diarization_check"]]
    final_df = revAI_check.merge(azure_speaker_number_df, on="filename", how="left")
    final_df = final_df.merge(word_count_df, on="filename", how="left")
    
    selected_df = asr_data[["filename", 'Azure', 'AWS', 'GoogleChirp', 'GoogleTelephony', 
                           'GoogleLong', 'Whisper', 'AssemblyAI', 'RevAI']]
    output_df = final_df.merge(selected_df, on="filename", how="left")
    
    interviewer_check_df = pd.DataFrame()
    if args.manual_check:
        print(f"Processing manual check file: {args.manual_check}")
        interviewer_check_df = pd.read_csv(args.manual_check)
        interviewer_check_df = interviewer_check_df[["filename", "manual_check", "no_tag"]]
        final_df = output_df.merge(interviewer_check_df, on="filename", how="left")
    else:
        final_df = output_df
        
    create_summary_stats(final_df, interviewer_check_df)
    filtered_df = filter_files(final_df, interviewer_check_df)
    
    date = datetime.datetime.now().strftime("%Y-%m-%d")
    output_path = args.output
    
    if not output_path.endswith('.csv'):
        if os.path.isdir(output_path):
            output_path = os.path.join(output_path, f"{date}_filtered_interviewer_speech.csv")
        else:
            output_path = f"{output_path}_{date}.csv"
    
    print(f"Writing {len(filtered_df)} records to {output_path}")
    filtered_df.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()