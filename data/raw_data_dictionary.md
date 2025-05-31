# Descriptions of variables in merged raw data file 
This readme file serves to explain variables included in the [Merged_wer_data_raw.csv](https://github.com/koenecke/aphasia_asr_audit/blob/main/data/Merged_WER_data_raw.csv.zip) data file. 

## Basic information columns
- segment_name: segment identification for each audio snippet
- Participant_ID: unique ID for participant 
- Group: which group each segment belongs to (aphasia vs control) 

## Ground Truth Versions 
Different versions of ground truth are included in the following columns:
- groundtruth: original ground truth from Aphasiabank 
- groundtruth_RF: cleaned ground truth that removes fillers (RF)
- groundtruth_RFF: cleaned ground truth that removes fillers and fragments  (RFF)
- groundtruth_RFFR: cleaned ground truth that removes fillers, fragments, and repeated single words (RFFR)
- groundtruth_RFFRR: cleaned ground truth that removes fillers and fragments, repeated single words, and repeated phrases (RFFRR)

## Word Count for Different Versions of Ground Truth 
Word count for each cleaned version of ground truth is being documented in the following columns:
- word_count_RF
- word_count_RFF  
- word_count_RFFR  
- word_count_RFFRR  

## ASR Transcript Version
Different cleaned versions (RF,RFF,RFFR,RFFRR) of transcripts from ASR systems are being included in the following columns. The format of names follows '[ASR]'+'_'+'Version of Cleaning'.
- AWS_orig, AWS_RF, AWS_RFF, AWS_RFFR, AWS_RFFRR, 
- Azure_orig, Azure_RF, Azure_RFF,Azure_RFFR, Azure_RFFRR  
- GoogleChirp_orig, GoogleChirp_RF, GoogleChirp_RFF, GoogleChirp_RFFR,GoogleChirp_RFFRR,
- RevAI_orig, RevAI_RF, RevAI_RFF, RevAI_RFFR, RevAI_RFFRR  
- Whisper_orig, Whisper_RF, Whisper_RFF,Whisper_RFFR,Whisper_RFFRR  
- AssemblyAI_orig, AssemblyAI_RF, AssemblyAI_RFF, AssemblyAI_RFFR, AssemblyAI_RFFRR  

## WER version 
WER for each snippet is calculated based on versions of ground truth and ASR transcripts. The format of names follows '[ground truth cleaned version]' + '_' + '[ASR cleaning version]'.

- original_groundtruth_AWS_orig_wer  
- original_groundtruth_Azure_orig_wer  
- original_groundtruth_AssemblyAI_orig_wer  
- original_groundtruth_GoogleChirp_orig_wer  
- original_groundtruth_RevAI_orig_wer  
- original_groundtruth_Whisper_orig_wer  
- groundtruth_RF_Whisper_RF_wer  
- groundtruth_RF_GoogleChirp_RF_wer  
- groundtruth_RF_RevAI_RF_wer  
- groundtruth_RF_AWS_RF_wer  
- groundtruth_RF_Azure_RF_wer  
- groundtruth_RF_AssemblyAI_RF_wer  
- groundtruth_RFF_Whisper_RFF_wer  
- groundtruth_RFF_GoogleChirp_RFF_wer  
- groundtruth_RFF_RevAI_RFF_wer  
- groundtruth_RFF_AWS_RFF_wer  
- groundtruth_RFF_Azure_RFF_wer  
- groundtruth_RFF_AssemblyAI_RFF_wer  
- groundtruth_RFFR_Whisper_RFFR_wer  
- groundtruth_RFFR_GoogleChirp_RFFR_wer  
- groundtruth_RFFR_RevAI_RFFR_wer  
- groundtruth_RFFR_AWS_RFFR_wer  
- groundtruth_RFFR_Azure_RFFR_wer  
- groundtruth_RFFR_AssemblyAI_RFFR_wer  
- groundtruth_RFFRR_Whisper_RFFRR_wer  
- groundtruth_RFFRR_GoogleChirp_RFFRR_wer  
- groundtruth_RFFRR_RevAI_RFFRR_wer  
- groundtruth_RFFRR_AWS_RFFRR_wer  
- groundtruth_RFFRR_Azure_RFFRR_wer  
- groundtruth_RFFRR_AssemblyAI_RFFRR_wer  

## Audio information
Information of each audio snippet is recorded in the following columns:
- total_audio_duration: total duration of audio snippet
- Duration_Type: binary variable that tracks if each audio snippet has a total duration length greater than 2 seconds
- nonvocal_audio_duration: total duration of a audio snippet that has no vocal information (calculated using Silero)
- total_audio_duration_group: group that each audio snippet falls into based on total audio duration 
- non_vocal_percentage: percentage of non-vocal duration in each audio snippet out of total duration
- Mean_Background_Noise: average background noise of each audio snippet
- Mean_Background_Noise_Levels: noise level group each audio snippet falls into 
- volume_level: overall volume of audio snippet

## Hallucination data 
Whisper_hallucination: binary variable that identifies whether hallucination is detected for each audio snippet in Whisper transcription
Whisper_hallucination_category: category of hallucination types each audio snippet falls into if hallucination is detected

## Demographic information
Demographic information of participants from AphasiaBank is included in the following columns 
- Age at Testing  
- Gender  
- Race  
- Primary Language  
- Language Status  
- Years of Education  
- Employment  Status  
- Adequate Vision  
- Adequate Hearing  
- Aphasia Duration  
- Aphasia Category -- Clin Impression  
- Aphasia Type --Clin Impression -- Boston  

- *aphasia_wab_score*: the Aphasia Quotient (AQ) score on the Western Aphasia Battery (WAB) to measure severity of aphasia for participants. 100 for control participants, indicating no severity. 



