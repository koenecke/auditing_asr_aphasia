# Addressing Pitfalls in Auditing Practices of Automatic Speech Recognition Technologies: A Case Study of People with Aphasia


## Getting Started
- Obtain API token for each Automated Speech Recognition system
- Obtain data access permission from Aphasia Bank based on their [data policy](https://talkbank.org/share/rules.html)

### Information collection of audio snippets 
- Use Python scripts in **src/ASR** folder to obtain **audio snippets** for testing ASR systems
- Use  Python scripts in **src/Ground Truth Transcripts Collection** folder to preprocess ground truth data from Aphasia Bank
- Utilize [noise_level.py](https://github.com/koenecke/auditing_asr_aphasia/blob/main/src/audio_measure/noise_level.py) to generate general background noise level of the audio file, using librosa.
- Utilize [volume_level.py](https://github.com/koenecke/auditing_asr_aphasia/blob/main/src/audio_measure/volume_level.py) to generate general volume level of the audio file, using librosa.
- Utilize [nonvocal_pyannote.py](https://github.com/koenecke/auditing_asr_aphasia/blob/main/src/audio_measure/nonvocal_pyannote.py), [nonvocal_silero.py](https://github.com/koenecke/auditing_asr_aphasia/blob/main/src/audio_measure/nonvocal_silero.py), or [nonvocal_webrtcvad.py](https://github.com/koenecke/auditing_asr_aphasia/blob/main/src/audio_measure/nonvocal_webrtcvad.py) to generate non-vocal duration of the audio file, using different VAD calculation packages.

### Collection of ASR transcripts

- Use Python scripts in **src/ASR** folder to obtain transcription from the six ASRs:
  
- Each script outputs a json files that contain the filenames and transcripts for audio files.
```
output = "DATE_[ASR]_transcript.json"
```

### Transcript Cleaning and WER Calculation
After obtaining transcripts from each ASR, 
- Utilize [transcript_cleaning.py](https://github.com/koenecke/auditing_asr_aphasia/blob/main/src/transcript_cleaning.ipynb) to
  - Merge ASR transcripts and ground truth transcript
  - Generate different cleaned versions of ground truth and ASR transcript 
  - Compute Word Error Rate (WER) results for individual audio file
  - Output files that contain all transcript results and WER rates
```
output = "[DATE]_WER_Results.csv"
```
### Data Processing and Analysis for WER data
Data processing in the following sequence:
- Utilize the [data_merge.rmd](https://github.com/koenecke/auditing_asr_aphasia/blob/main/src/Analysis/data_merge.Rmd) to merge transcripts, ground truth, WER results, and information related to audio measure and demographic information. The zipped version of this data is stored [here](https://github.com/koenecke/auditing_asr_aphasia/blob/main/data/Merged_WER_data_raw.csv.zip) since the original data file is too big. 
```
output = "Merged_WER_data_raw.csv"
```
*Descriptions of variables for this file are documented in [raw_data_dictionary.md](https://github.com/koenecke/aphasia_asr_audit/blob/main/data/raw_data_dictionary.md)* 
- Utilize the [data.preprocessing.rmd](https://github.com/koenecke/auditing_asr_aphasia/blob/main/src/Analysis/data_processing.R) to clean data columns
- Utilize the [matching.R](https://github.com/koenecke/auditing_asr_aphasia/blob/main/src/Analysis/matching.R) to perform matching for balanced subsets

To calculate weighted average of WER for each group,
- Utilize [calculate_weighted_WER.py](https://github.com/koenecke/auditing_asr_aphasia/blob/main/src/calculate_weighted_WER.py)

For analysis of WER across ASR:
- Utilize [analysis.rmd](https://github.com/koenecke/auditing_asr_aphasia/blob/main/src/Analysis/analysis.Rmd) to generate figures and regression (figures: 2,S4-6; Tables: S2-8)  
- Utilize [weighted_average_analysis.rmd](https://github.com/koenecke/auditing_asr_aphasia/blob/main/src/Analysis/weighted_average_analysis.Rmd) to generate figures for weighted average (figures:3,4,S1-3;Tables: S2-6) 

### ASR metric suite calculation
In order to look at metrics other than WER (BLEU, ROUGE-1, ROUGE-2, ROUGE-L, METEOR, Insertion rate, WIL, RIL, CER):
- Utilize the [nlp_metric.py](https://github.com/koenecke/auditing_asr_aphasia/blob/main/src/Analysis/nlp_metric.py) to
    - Calculate each metric results
    - Filter to hallucination candidates subject to manual review

### Hallucination experiment
- The entire list of audio samples used for the hallucination experiment is stored in [experiment_wer_rates_2024-07-03.csv](https://github.com/koenecke/auditing_asr_aphasia/blob/main/data/Hallucination_Experiment/experiment_wer_rates_2024-07-03.csv). This file contains the groundtruth, Whisper and Google Chirp transcripts of each audio variation, and their hallucination information. Check [audio_experiment_samples_cut.csv](https://github.com/koenecke/auditing_asr_aphasia/blob/main/data/Hallucination_Experiment/audio_experiment_samples_cut.csv) for specific time stamp information used to cut the audio samples midway. Real life noise audio file used for the analysis was [360703__eguobyte__large_crowd_medium_distance_stereo.wav](https://freesound.org/people/eguobyte/sounds/360703/)
- Utilize the [experiment_audio_manipulation.py](https://github.com/koenecke/auditing_asr_aphasia/blob/main/src/Analysis/hallucination/experiment_audio_manipulation.py) to
  - Generate variations of each audio file
  - Check for statistical significance using [hallucination_experiment.py](https://github.com/koenecke/auditing_asr_aphasia/blob/main/src/Analysis/hallucination/hallucination_experiment.py) (figures: S7; Tables: 3)
