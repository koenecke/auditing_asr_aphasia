import pandas as pd
import jiwer
import numpy as np


def compute_wer(ref, hyp):
    transformation = jiwer.Compose([
        jiwer.RemovePunctuation(),
        jiwer.ToLowerCase(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip()
    ])
    ref = transformation(ref)
    hyp = transformation(hyp)
    return jiwer.wer(ref, hyp)

def compute_editdistance(ref, hyp):
    transformation = jiwer.Compose([
        jiwer.RemovePunctuation(),
        jiwer.ToLowerCase(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip()
    ])
    ref = transformation(ref)
    hyp = transformation(hyp)
    measures = jiwer.compute_measures(ref, hyp)
    
    insertions = measures['insertions']
    deletions = measures['deletions']
    substitutions = measures['substitutions']
    ref_len = len(ref.split())
    edit_ops = insertions + deletions + substitutions
    
    return edit_ops, ref_len, insertions, deletions, substitutions


def calculate_weighted_average_WER_by_group(data, group = 'Group', cleaning_version = 'V3'):

    api_columns = ['AWS', 'Azure', 'GoogleChirp', 'RevAI', 'Whisper', 'AssemblyAI']

    api_wer_columns = [f'{api}_{cleaning_version}' for api in api_columns]
    print(api_wer_columns)
    results = []

    for subgroup in data[group].unique():
        print(subgroup)
        if pd.isnull(subgroup):
            continue
        group_data = data[data[group] == subgroup]
        print(f"group data length {len(group_data)}")
        for api in api_wer_columns:
            total_wer = 0
            total_words = 0
            total_insertions = total_deletions = total_substitutions = 0
            

            # store WER for each sample in the group
            wer_list = []
            # store groundtruth word count for each sample in the group
            word_count_list = []


            for index, row in group_data.iterrows():

                # set groundtruth version to compare with
                ref = row[f'groundtruth_{cleaning_version}']
                hyp = row[api]
                
                if isinstance(ref, str) and isinstance(hyp, str):
                    wer = compute_wer(ref, hyp)
                    total_wer += wer

                    edit_ops, ref_len, insertions, deletions, substitutions = compute_editdistance(ref, hyp)
                    total_words += ref_len
                    total_insertions += insertions
                    total_deletions += deletions
                    total_substitutions += substitutions

                    wer_list.append(wer)
                    word_count_list.append(ref_len)
            
            standard_wer = total_wer / len(group_data)
            weighted_wer = (total_insertions + total_deletions + total_substitutions) / total_words
            print(f"subgroup: {subgroup}, api: {api}, standard WER: {standard_wer}, weighted WER: {weighted_wer}")
            weighted_wer2 = np.sum(np.array(word_count_list) * np.array(wer_list)) / np.sum(word_count_list)
            # print(f"weighted wer 1: {weighted_wer}, weighted wer 2: {weighted_wer2}")

            # calculate standard error for weighted WER 
            n = len(wer_list)
            if n > 1:

                ## calculate this based on https://stackoverflow.com/questions/61831353/how-can-i-calculate-weighted-standard-errors-and-plot-them-in-a-bar-plot
                weighted_wer_variance = np.sum( np.array(word_count_list) * (np.array(wer_list)-weighted_wer) **2) / (np.sum(word_count_list)-1)
                weighted_wer_sd = np.sqrt(weighted_wer_variance * n / (n-1))
                weighted_wer_se = weighted_wer_sd / np.sqrt(n)

                # check another way of calculating SD
                weighted_wer_variance_pre = np.sum( np.array(word_count_list) * (np.array(wer_list)-weighted_wer2) **2) 

                ## obtain number of nonzeros in word_count_list
                n_nonzero = np.count_nonzero(word_count_list)
                ## calculate this based on https://stats.stackexchange.com/questions/6534/how-do-i-calculate-a-weighted-standard-deviation-in-excel
                weighted_wer_sd2 = np.sqrt(weighted_wer_variance_pre / ((n_nonzero-1)/n_nonzero * np.sum(word_count_list)))
                weighted_wer_se2 = weighted_wer_sd2 / np.sqrt(n)
                # print(f"weighted wer sd 1: {weighted_wer_sd}, weighted wer sd 2: {weighted_wer_sd2}")
                # print(f"weighted wer se 1: {weighted_wer_se}, weighted wer se 2: {weighted_wer_se2}")
         
                # print (round(weighted_wer_se,ndigits=4) == round(weighted_wer_se2,ndigits=4))

            else:   
                weighted_wer_se = 0


            results.append({
                'Group': subgroup,
                'API': api,
                'Standard WER': standard_wer,
                'Weighted WER': weighted_wer,
                'Weighted WER SD': weighted_wer_sd,
                'Weighted WER SE': weighted_wer_se
            })
    results = pd.DataFrame(results)
    return results


# write a main function that takes argument matched or unmatched
def main(matched=True):
    global MATCHED
    MATCHED = matched
    # rest of the code will run here

    file_path = '../data/Merged_WER_data_raw_cleaned.csv'

    full_data = pd.read_csv(file_path)


    # set output filename based on matching
    if MATCHED:
        output_filepath = '../data/weighted_average_WER_by_group_matched.csv'
    else:
        output_filepath = '../data/weighted_average_WER_by_group_unmatched.csv'


    # matched on aphasia vs control 
    file_path = '../data/matched_segment.csv'
    matched_data = pd.read_csv(file_path, low_memory=False)
    filter_data = full_data[full_data['segment_name'].isin(matched_data['segment_name'])]

    # matched on control vs fluent aphasia vs non-fluent aphasia
    file_path= "../data/three_way_matched_subset.csv"
    three_way_matched_data = pd.read_csv(file_path, low_memory=False)
    filtered_data3 = full_data[full_data['segment_name'].isin(three_way_matched_data['segment_name'])]


    if MATCHED:
        data = filter_data
    else:
        data = full_data

    v3_results_df = calculate_weighted_average_WER_by_group(data=data,group = 'Group', cleaning_version = 'RFFRR')

    v3_results_df['weighted_average_group']='aphasia_type'

        
    # Aphasia type boston
    selected_aphasia_type = ["None","Anomic","Conduction","Wernicke","Broca","Global"]
    this_data = data[data['aphasiaTypeBoston'].isin(selected_aphasia_type)]

    v3_results_aphasia_type2 = calculate_weighted_average_WER_by_group(data = this_data,group = 'aphasiaTypeBoston', cleaning_version = 'RFFRR')
    v3_results_aphasia_type2['weighted_average_group']='aphasiaTypeBoston'

            
    # run this chunk only if the data is matched on aphasia category 
    if MATCHED:
        data = filtered_data3 # matched three way data
    else:
        data = full_data
    # Aphasia type 2
    v3_results_aphasia_type = calculate_weighted_average_WER_by_group(data = data,group = 'aphasia_TypeFluency', cleaning_version = 'RFFRR')
    v3_results_aphasia_type['weighted_average_group']='aphasia_TypeFluency'



    # merge all the results
    all_results = pd.concat([v3_results_df,v3_results_aphasia_type,v3_results_aphasia_type2])

    # add asr name by splitting the string of the API column
    all_results['ASR'] = all_results['API'].str.split('_').str[0]
    all_results.to_csv(output_filepath, index=False)  

if __name__ == "__main__":
    print("----matching version-----")
    main(matched=True)
    print("----unmatching version-----")
    main(matched=False)