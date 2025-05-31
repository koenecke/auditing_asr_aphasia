# Perform matching for the full dataset 


cleaned_data <- read_csv("../../data/Merged_WER_data_raw_cleaned.csv")


## Select smaller data frame to save space 
subset_data_for_matching <- cleaned_data %>%
  dplyr::select(segment_name,
                is_aphasia,
                fluent_aphasia,
                aphasiaTypeBoston,
                is_female,
                age, 
                edu_years,
                race_fac,
                employ_fac,
                word_count_RFFRR, 
                Mean_Background_Noise) 

################################# Match between aphasia and control #################################################

## Propensity score matching
perform_matching <- function(df){
  set.seed(100)
  # # select matching covariates 
  this.m.out <- matchit(is_aphasia ~ age + edu_years + race_fac  +
                          is_female  + word_count_RFFRR + Mean_Background_Noise,
                        data = df,
                        distance ="glm",
                        caliper= 0.13,
                        method = "nearest",
  )
  
  # plot propensity scores
  # plot(this.m.out, type = "jitter", interactive = FALSE)
  
  p <- love.plot(bal.tab(this.m.out),
                 stat = "mean.diffs",
                 threshold = .1,
                 drop.distance = TRUE,
                 var.order = "unadjusted",
                 abs = TRUE,
                 line = TRUE)
  print(p)
  # 
  print(summary(this.m.out))
  
  # subset to matched snippets
  
  matched_df <- match.data(this.m.out)
  
  return(matched_df)
}


# produce matched subset  
post_matching_segment_data <- subset_data_for_matching %>% 
  perform_matching() %>% dplyr::select(segment_name,is_aphasia,aphasiaTypeBoston,fluent_aphasia) 

# overview of count for aphasia and control post matching
post_matching_segment_data %>% group_by(is_aphasia) %>% count()

post_matching_segment_data %>% group_by(aphasiaTypeBoston) %>% count()

write_csv(post_matching_segment_data %>% select(segment_name),"../../data/matched_segment.csv")


##########################Matching between fluent vs nonfluent vs control##########################


## Perform three-way matching (common-referent approach)

# step 1: count control, fluent aphasia, nonfluent aphasia--> control has the smallest sample size
subset_data_for_matching %>% 
  filter(!is.na(fluent_aphasia)) %>% # filter out aphasia that has no clinical impression (fluent/nonfluent)
  dplyr::select(segment_name,is_aphasia,fluent_aphasia) %>% 
  unique() %>% 
  group_by(is_aphasia,fluent_aphasia) %>% 
  count()

# step 2: obtain dataset for matching
## dataset 1: only fluent aphasia and control samples
fluent_control_df <- subset_data_for_matching %>%  filter(is_aphasia==0 | fluent_aphasia==1)
fluent_control_df %>% group_by(is_aphasia,fluent_aphasia) %>% count()

## dataset 2: only nonfluent aphasia and control samples 
nonfluent_control_df <- subset_data_for_matching %>% filter(is_aphasia==0 | (is_aphasia==1 & fluent_aphasia==0))
nonfluent_control_df %>% group_by(is_aphasia,fluent_aphasia) %>% count()


set.seed(100)
# step 3: first match between fluent vs control 
this.m.out <- matchit(is_aphasia~ age + edu_years + race_fac + is_female,
                      data = fluent_control_df ,
                      distance ="glm",
                      method = "nearest",
                      caliper=0.15
)


p <- love.plot(bal.tab(this.m.out),
               stat = "mean.diffs",
               threshold = .1,
               drop.distance = TRUE,
               var.order = "unadjusted",
               abs = TRUE,
               line = TRUE)
print(p)

print(summary(this.m.out))
fluent_control_matched_df <- match.data(this.m.out)

## obtain fluent aphasia matched subset 
fluent_subset <- fluent_control_matched_df %>% filter(is_aphasia==1) %>% dplyr::select(segment_name)

set.seed(10)

# step 4: next match between nonfluent vs control
this.m.out <- matchit(is_aphasia~ age + edu_years + race_fac  + is_female,
                      data =nonfluent_control_df ,
                      distance ="glm",
                      method = "nearest",
                      caliper=0.15
)


p <- love.plot(bal.tab(this.m.out),
               stat = "mean.diffs",
               threshold = .1,
               drop.distance = TRUE,
               var.order = "unadjusted",
               abs = TRUE,
               line = TRUE)
print(p)

print(summary(this.m.out))
nonfluent_control_matched_df <- match.data(this.m.out)

##  obtain nonfluent aphasia matched subset 
nonfluent_subset <- nonfluent_control_matched_df %>% filter(is_aphasia==1) %>% dplyr::select(segment_name)


# step 5 take common subset of control # take common subset of control segment_name
overlap_control_subset <- fluent_control_matched_df %>% dplyr::select(segment_name) %>% mutate(fluent_control=1) %>% 
  full_join(nonfluent_control_matched_df %>% dplyr::select(segment_name) %>% mutate(nonfluent_control=1)) %>% 
  filter(fluent_control==1 & nonfluent_control==1) %>% dplyr::select(segment_name)


# step 6: identify common subset 
three_way_subset <- cleaned_data %>% filter(segment_name %in% nonfluent_subset$segment_name |segment_name %in% fluent_subset$segment_name |segment_name %in% overlap_control_subset$segment_name)

write_csv(three_way_subset,"../../data/three_way_matched_subset.csv")

# Overview of sample size for each group after three- way matching 
three_way_subset %>% dplyr::select(aphasia_TypeFluency,segment_name) %>% unique() %>%  group_by(aphasia_TypeFluency) %>% count()
