import os
import re
import pylangacq as pla
import pandas as pd
from datetime import datetime

current_date = datetime.now().strftime('%Y-%m-%d')
print(current_date)

# read in chat file
# the below function will skip the utterance where the error is occurring - no need to clean manually
def read_chat_file(file_path):
    try:
        ds = pla.read_chat(file_path)

        cols = ['start_time', 'end_time']
        lst = []
        raw_transcript = []

        for utterance in ds.utterances(participants="PAR"):
            try:
                time_marks = utterance.time_marks
                if time_marks:
                    lst.append(time_marks)
                    raw_transcript.append(utterance.tiers['PAR'][:-16])
            except ValueError as e:
                print(f"Skipped problematic utterance in file: {file_path}. Error message: {e}")

        if not lst:  # No valid utterances found
            return None

        df = pd.DataFrame(lst, columns=cols)
        df = df.assign(raw_transcript=raw_transcript)
        df['filename'] = os.path.basename(file_path)

        return df
    except Exception as e:
        print(f"Error processing file: {file_path}. Error message: {e}")
        return None
    
def read_all_chat_files_and_save(input_directory, output_csv_name):
    all_data = []  
    
    for root, dirs, files in os.walk(input_directory):
        for file in files:
            if file.endswith(".cha") and '-checkpoint' not in file:
                file_path = os.path.join(root, file)
                df = read_chat_file(file_path)
                if df is not None:
                    all_data.append(df)
    
    final_df = pd.concat(all_data, ignore_index=True)
    final_df.to_csv(output_csv_name, index=False)
    print(f"Saved to {output_csv_name}")

    return final_df

read_all_chat_files_and_save('../../../Aphasia_transcript/', f'../../data/aphasia_all_{current_date}.csv') 
read_all_chat_files_and_save('../../../Control_transcript/', f'../../data/control_all_{current_date}.csv')

def add_segment_names(df, original_filename):
    
    df['file_base'] = df['filename'].str.split('.').str[0]
    df['segment_name'] = df['file_base'] + "_" + df['start_time'].astype(str) + "_" + df['end_time'].astype(str) + '.wav'
    df.drop('file_base', axis=1, inplace=True)
    df.to_csv(original_filename, index=False)

    return df

aphasia_df = pd.read_csv(f"../../data/aphasia_all_{current_date}.csv")
control_df = pd.read_csv(f"../../data/control_all_{current_date}.csv")
aphasia_df = add_segment_names(aphasia_df, f"../../data/aphasia_all_{current_date}.csv")
control_df = add_segment_names(control_df, f"../../data/control_all_{current_date}.csv")

def clean_transcription(transcription):
        
    transcription = transcription.replace('&-', 'FILLER')
    transcription = transcription.replace('&+', 'FRAGMENT')
    
    transcription = transcription.replace('[<]', '').replace('[.]', '').replace('[>]', '')
    transcription = transcription.replace('[>1]', '').replace('[>2]', '').replace('[>3]', '').replace('[<1]', '').replace('[<2]', '').replace('[<3]', '')
    transcription = transcription.replace('+<', '').replace('+,', '').replace('+..?', '').replace('+..', '').replace('+/?', '').replace('[?]', '')
    transcription = transcription.replace('(.)', '').replace('(..)', '').replace('(...)', '')
    transcription = transcription.replace('<', '').replace('>', '')
    transcription = transcription.replace('(', '').replace(')', '')
    transcription = transcription.replace('‡', '').replace('[/]', '').replace('[//]', '').replace('[///]', '').replace('[/?]', '').replace('[/-]', '')
    
    # Remove [+ ~]
    transcription = transcription.replace('[+ gram]', '').replace('[+ gram', '').replace('[+ gra', '').replace('[+ gr', '').replace('[+ g', '')
    transcription = transcription.replace('[+ exc]', '').replace('[+ exc', '').replace('[+ ex', '')
    transcription = transcription.replace('[+ esc]', '').replace('[+ esc', '').replace('[+ es]', '').replace('[+ es', '').replace('[+ e', '')
    transcription = transcription.replace('[+ jar]', '').replace('[+ per]', '').replace('[+ jar', '').replace('[+ ja', '').replace('[+ j', '')
    transcription = transcription.replace('[+ circ]', '').replace('[+ cir]', '').replace('[+ cir', '').replace('[+ ci', '').replace('[+ c', '')
    transcription = transcription.replace('[+ ', '').replace('[+', '')
    
    # Replace xxx
    transcription = transcription.replace('xxx', 'UNK')
    
    # Error words
    # on [: and] [* p:w] -> on
    # transcription = re.sub(r"\s([a-zA-Z'_]+)\s\[\:\s([a-zA-Z'_\s]+)\]\s\[\*\s([a-zA-Z'_]*\:[a-z-]*)\]", r' \1', transcription)
    transcription = re.sub(r"(?:^|\s)([a-zA-Z'_-]+)\s\[\:\s([a-zA-Z'\s_-]+)\]\s\[\*\s*([*:a-zA-Z\d\s'+-=]*)\]", r' \1', transcription)
    
    # ain't [: are not] (without error code) -> ain't
    transcription = re.sub(r"(?:^|\s)([a-zA-Z'_-]+)\s\[\:\s([a-zA-Z'\s_-]+)\](?!\s\[)", r' \1', transcription)
    
    # honli@u [: only] [* p:n] -> honli
    transcription = re.sub(r"(?:^|\s)([a-zA-Z'_-]+)@u\s\[\:\s([a-zA-Z'_@\s]*)\]\s\[\*\s*([*:a-zA-Z\d\s'+-=]*)\]", r' \1', transcription)
        
    # kotəgəl@u [: comfortable] [* n:k] -> UNK
    # hɑspəlɪd@u [: hospital] [* n:k-ret] -> UNK
    # ðæɾɪ@u [: x@n] [* n:uk] -> UNK
    # mɔ@u [: x@n] [* n:uk-rep] -> UNK
    # fɪŋks@u [: sphinx] [*] -> UNK
    transcription = re.sub(r"(\S*@u)\s\[\:\s([a-zA-Z'_@\s]*)\]\s\[\*\s*([*:a-zA-Z\d\s'+-=]*)\]", r'UNK', transcription)

    # ʌt [: up] [* p:n] -> UNK
    # iʔi [: x@n] [* n:uk] -> UNK
    transcription = re.sub(r"([a-zA-Z'_]*[^a-zA-Z'_\s]+[a-zA-Z'_]*[^a-zA-Z'_\s]*[a-zA-Z'_]*[^a-zA-Z'_\s]*[a-zA-Z'_]*[^a-zA-Z'_\s]*[a-zA-Z'_]*[^a-zA-Z'_\s]*[a-zA-Z'_]*[^a-zA-Z'_\s]*[a-zA-Z'_]*[^a-zA-Z'_\s]*)\s\[\:\s([a-zA-Z'_@\s]+)\]\s\[\*\s*([*:a-zA-Z\s'+-=]*)\]", r'UNK', transcription)
    transcription = re.sub(r"\S*\s\[\:\s([a-zA-Z'_@$\s]+)\]\s\[\*\s*([*:a-zA-Z\s'+-=]*)\]", r'UNK', transcription)
                 
    # sɪnrɛlə@u [: Cinderella] (without error code) -> UNK
    # transcription = re.sub(r"(\S*@u)\s\[\:\s([a-zA-Z'_@\s]*)\](?!\s\[\*)", r'UNK', transcription)
    transcription = re.sub(r"([a-zA-Z'_]*[^a-zA-Z'_\s]+[a-zA-Z'_]*[^a-zA-Z'_\s]*[a-zA-Z'_]*[^a-zA-Z'_\s]*[a-zA-Z'_]*[^a-zA-Z'_\s]*[a-zA-Z'_]*[^a-zA-Z'_\s]*[a-zA-Z'_]*[^a-zA-Z'_\s]*[a-zA-Z'_]*[^a-zA-Z'_\s]*)\s\[\:\s([a-zA-Z'_@\s]+)\]", r'UNK', transcription)
    
    # walked [: s:uk-ret] (without replacement, error code typo) -> walked
    transcription = re.sub(r"(?:^|\s)([a-zA-Z'_-]+)\s\[\:\s([a-zA-Z':\s-]+)\](?!\s\[\*)", r' \1', transcription)
    
    # remaining error codes
    transcription = re.sub(r'\[\*\s[^\]]+\]', '', transcription)
    transcription = transcription.replace('[*]', '')
    
    # untackled cases
    transcription = re.sub(r"(\S*@u)\s\[\:\s(\S*)\]", 'UNK', transcription)
    transcription = re.sub(r"([a-zA-Z'_]*(?!@)[^a-zA-Z'_\s]+[a-zA-Z'_]*)\s\[\:\s([a-zA-Z'_@\s]+)\]", 'UNK', transcription)
    
    
    # start of the string
    # transcription = re.sub(r"([a-zA-Z']+)\s\[:\s([a-zA-Z']+)\]", r'\1', transcription)
                           
    # Error codes
    transcription = re.sub(r'\$\S+', '', transcription)  
    transcription = re.sub(r'@l', '', transcription)
    transcription = re.sub(r'@o', '', transcription)
    transcription = re.sub(r'@b', '', transcription)
    transcription = re.sub(r'@q', '', transcription)
    transcription = re.sub(r'@k', '', transcription)
    transcription = re.sub(r'@i', '', transcription)
    # transcription = re.sub(r'@n', '', transcription) 
    transcription = re.sub(r'@si', '', transcription)
    
    transcription = re.sub(r'([a-zA-Z]+)@n', r'\1', transcription)
    transcription = re.sub(r'\S+@u', r'UNK', transcription)

    # INV
    transcription = re.sub(r'&\*INV\S+', '', transcription)
    
    # Remove words with &-, &+, &=, [=! ], [= ], [!], [% ]
    # need to include & because typo
    #transcription = re.sub(r'&[+-]?.*?\s', '', transcription)
    transcription = re.sub(r'&\S*', '', transcription)
    transcription = re.sub(r'&[^ ]*', '', transcription)
    # transcription = re.sub(r'\[=!\s[a-zA-Z]*\]', '', transcription)
    transcription = transcription.replace('[!]', '')
    transcription = re.sub(r'\[=[^\]]+\]', '', transcription)
    transcription = transcription.replace('[=! laughin', '')
    transcription = re.sub(r"\[%\s(.*)\]", '', transcription)
    
    # Remove unnecessary chars
    transcription = transcription.replace('+', '').replace('"', '').replace('...', '').replace('//', '').replace('/', '').replace('^', '').replace('„', '')
    
    # Replace _ with space
    transcription = transcription.replace('_', ' ')

    # Remove punctuation
    transcription = transcription.replace('.', ' ')
    transcription = transcription.replace('?', ' ')
    transcription = transcription.replace('!', ' ')
    transcription = transcription.replace('”', '').replace('“', '')    
     
    # Remove :
    transcription = transcription.replace(':', '')
    
    # Replace - with whitespace
    transcription = transcription.replace('-', ' ')
    
    # Remove words containing '0' - there's error code with '0
    transcription = ' '.join(word for word in transcription.split() if '0' not in word)
    
    # Standardize whitespace
    transcription = re.sub(r'\s+', ' ', transcription)
    
    # Unknown - I suspect linebreak
    transcription = transcription.replace('', '')
    
    # transcription = transcription.replace('FILLER', '&-')
    # transcription = transcription.replace('FRAGMENT', '&+')
    
    return transcription.strip()
    
aphasia_df['clean'] = aphasia_df['raw_transcript'].apply(lambda x: clean_transcription(str(x)))
aphasia_df['clean_original'] = aphasia_df['clean']
aphasia_df.to_csv(f'../../data/aphasia_all_fix_{current_date}.csv', index=False)

control_df['clean'] = control_df['raw_transcript'].apply(lambda x: clean_transcription(str(x)))
control_df['clean_original'] = control_df['clean']
control_df.to_csv(f'../../data/control_all_fix_{current_date}.csv', index=False)

def clean_version1(text):
# fillers and phonological fragments are both left in

    text = str(text)
    return text.replace('FILLER', '').replace('FRAGMENT', '')

def clean_version2(text):
# fillers are removed, phonological fragments are left in

    text = str(text)
    text = ' '.join([word for word in text.split() if not word.startswith('FILLER')])
    text = text.replace('FRAGMENT', '')
    
    return text

def clean_version3(text):
# fillers and phonological fragments are both removed

    text = str(text)
    text = ' '.join([word for word in text.split() if not word.startswith('FRAGMENT') and not word.startswith('FILLER')])

    return text

aphasia_df['clean_v1'] = aphasia_df['clean'].apply(clean_version1)
aphasia_df['clean_v2'] = aphasia_df['clean'].apply(clean_version2)
aphasia_df['clean_v3'] = aphasia_df['clean'].apply(clean_version3)

control_df['clean_v1'] = control_df['clean'].apply(clean_version1)
control_df['clean_v2'] = control_df['clean'].apply(clean_version2)
control_df['clean_v3'] = control_df['clean'].apply(clean_version3)

aphasia_df.to_csv(f'../../data/aphasia_all_fix_{current_date}.csv')
control_df.to_csv(f'../../data/control_all_fix_{current_date}.csv')