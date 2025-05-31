# This file processes merged raw data and produce a cleaned data set for analysis 


library(R.utils)

# check if unzipped successfully
print("Is unzipped file detected: ")

file.exists("../../data/Merged_WER_data_raw.csv")

# See file extension details
wer_results_filepath = "../../data/Merged_WER_data_raw.csv"
post_demo_filter_data <- read_csv(wer_results_filepath)



colnames(post_demo_filter_data)
# Standardize names for columns 
cleaned_data <- post_demo_filter_data %>% 
  mutate(
    aphasiaTypeBoston = ifelse(`Aphasia Type --Clin Impression -- Boston`=="NA","Not Applicable",
                               ifelse(`Aphasia Type --Clin Impression -- Boston`=="ANO","Anomic",
                                      ifelse(`Aphasia Type --Clin Impression -- Boston`=="BRO","Broca",
                                             ifelse(`Aphasia Type --Clin Impression -- Boston`=="CON","Conduction",
                                                    ifelse(`Aphasia Type --Clin Impression -- Boston`=="GLO","Global",ifelse(`Aphasia Type --Clin Impression -- Boston`=="MTC","Mixed Transcortical",
                                                                                                                             ifelse(`Aphasia Type --Clin Impression -- Boston`=="TCM","Transcortical motor",
                                                                                                                                    ifelse(`Aphasia Type --Clin Impression -- Boston`=="TCS","Transcortical sensory",
                                                                                                                                           ifelse(`Aphasia Type --Clin Impression -- Boston`=="WER","Wernicke",
                                                                                                                                                  ifelse(`Aphasia Type --Clin Impression -- Boston`=="NCL","Non-classifiable",
                                                                                                                                                         ifelse(`Aphasia Type --Clin Impression -- Boston`=="OPT","Optic",
                                                                                                                                                                ifelse(`Aphasia Type --Clin Impression -- Boston`=="OTH","Other",
                                                                                                                                                                       ifelse(`Aphasia Type --Clin Impression -- Boston`=="U","Unavailable",NA))))))))))))),
    aphasia_TypeFluency = ifelse(`Aphasia Category -- Clin Impression` == "NFL","Non-fluent",
                                 ifelse(`Aphasia Category -- Clin Impression` == "FLU","Fluent",NA)),
    age =`Age at Testing`,
    age_group = cut(age, breaks = age_groups, labels = age_group_labels, include.lowest = TRUE),
    employ_fac = ifelse(`Employment  Status`=="R. W","W", `Employment  Status`),
    edu_years = round(as.numeric(`Years of Education`)),
    edu_levels = cut(edu_years, breaks = edu_groups, labels = edu_group_labels, include.lowest = TRUE),
    english_firstlang = ifelse(`Primary Language` %in% c("eng","English"), 1, 0),
    aphasia_duration = as.numeric(`Aphasia Duration`),
    
    ## standardize naming for aphasia category
    aphasiaTypeBoston = ifelse(Group=="control","control",aphasiaTypeBoston),
    aphasia_TypeFluency = ifelse(Group=="control","control",aphasia_TypeFluency),
    
    ## create dummy variable for fluent aphasia and nonfluent aphasia
    fluent_aphasia =as.factor(ifelse(aphasia_TypeFluency=="Fluent",1,ifelse(is.na(aphasia_TypeFluency),0,0))),
    nonfluent_aphasia =as.factor(ifelse(aphasia_TypeFluency=="Non-fluent",1,ifelse(is.na(aphasia_TypeFluency),0,0))),
    
    ## recategorize demo data based on their other clinical impression
    aphasia_TypeFluency = ifelse(is.na(aphasia_TypeFluency) & aphasiaTypeBoston %in%c("Broca","Transcortical motor","Mixed Transcortical","Global"),"Non-fluent",
                     ifelse(is.na(aphasia_TypeFluency) & aphasiaTypeBoston %in% c("Wernicke","Transcortical sensory","Conduction","Anomic"),"Fluent",
                            ifelse(Group =="control","control",aphasia_TypeFluency))),
    
     ## create dummy variable for other demo variables
     is_female = ifelse(Gender=='F',1,0),
    race_fac = ifelse(Race =="WH","White",ifelse(Race=="AA","African American","Other")),
    is_aphasia = ifelse(Group == 'aphasia',1,0),
    is_WH = ifelse(Race =="WH",1,0),
    is_AA = ifelse(Race =="AA",1,0),
    is_OTH = ifelse(race_fac =="Other",1,0)
    
  ) %>% select(-c(`Age at Testing`,`Employment  Status`,`Adequate Hearing`,`Adequate Vision`,`Years of Education`))

colnames(cleaned_data)

# count for each aphasia type (fluency)
cleaned_data %>% group_by(is_aphasia,fluent_aphasia) %>% count()
# count for each aphaisa type (Boston)
cleaned_data %>% group_by(is_aphasia,aphasiaTypeBoston) %>% count()

# Save cleaned raw data # Save cleaneaphasiaTypeBostond raw data 
write_csv(cleaned_data,"../../data/Merged_WER_data_raw_cleaned.csv")


# Preprocess raw data and convert it into long form for analysis 
word_count_group <-c(0,5,10,15,20,25,30,35,40,Inf)
word_count_group_labels <- c("0-5 words","5-10 words","10-15 words","15-20 words","20-25 words","25-30 words","30-35 words","35-40 words","above 40 words")


all_asr_list<-c("AWS","AssemblyAI","GoogleChirp","Azure","OpenAI","RevAI")

asr_list<-c("Amazon AWS","AssemblyAI","GoogleChirp","Azure","OpenAI","RevAI")


v1_name <- "Remove fillers"
v1plus_name <-"Remove fillers and fragments"
v2_name <- "Remove fillers, fragments,and repeated words"
v3_name<- "Remove fillers, fragments,repeated words, and repeated phrases"

# Removing columns that are needed for analysis to reduce file size 
pivoted_data <- cleaned_data %>% 
  select(-c(groundtruth,groundtruth_RF,groundtruth_RFF,groundtruth_RFFR,groundtruth_RFFRR, Whisper_orig,Whisper_RF,Whisper_RFF,Whisper_RFFR,Whisper_RFFRR)) %>% 
  pivot_longer(cols = contains("wer"), names_to = "WER_version", values_to = "WER") %>%
  dplyr::select(-contains(all_asr_list)) %>% 
  mutate(
    split_WER = str_split(WER_version, "_"),
    groundtruth_version = map_chr(split_WER, 2),
    asr_transcript_version = map_chr(split_WER, 4),
    ASR = map_chr(split_WER, 3),
    Word_Count_Group = cut(word_count_RF, breaks = word_count_group, labels = word_count_group_labels, include.lowest = TRUE),
    # Specific non-fluent aphasia syndromes include Broca, transcortical motor, mixed transcortical, and global. Fluent aphasia syndromes include Wernicke, transcortical sensory, conduction, and anomic.
  ) %>%
  dplyr::select(-c(WER_version, split_WER)) %>%
  filter(!ASR %in% c("WhisperDistil","GoogleLong","GoogleTelephony")) %>% 
  mutate(Duration_Type= ifelse(total_audio_duration<2,"<2 seconds",">=2 seconds"),
         ASR = case_when(ASR =="AWS"~"Amazon AWS",
                         ASR =="GoogleChirp"~"Google Chirp",
                         ASR =="Whisper"~"OpenAI Whisper",
                         ASR =="RevAI"~"Rev AI",
                         ASR =="Azure"~"Microsoft Azure",
                         ASR =="AssemblyAI"~"AssemblyAI"),
         Transcript_Version = case_when (
           asr_transcript_version =="orig"~"Original",
           asr_transcript_version =="RF"~v1_name,
           asr_transcript_version =="RFF"~v1plus_name,
           asr_transcript_version =="RFFR"~ v2_name,
           asr_transcript_version=="RFFRR"~v3_name))

print("Checking column names of processed data frame...")
colnames(pivoted_data)


write_csv(pivoted_data,"../../data/Post_Processing_WER_Data.csv")

