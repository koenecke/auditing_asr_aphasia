import pandas as pd
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from jiwer import cer, compute_measures, wil
from rouge_score import rouge_scorer

df = pd.read_csv('../../data/Merged_WER_data_raw_6k.csv')

services = ['RevAI', 'GoogleChirp', 'Whisper', 'AWS', 'Azure', 'AssemblyAI']
version = 'RFFRR'

for service in services:
    df[f'{service}_{version}'] = df[f'{service}_{version}'].astype(str)
df[f'groundtruth_{version}'] = df[f'groundtruth_{version}'].astype(str)

def calculate_bleu(reference, hypothesis):
    if pd.isna(hypothesis):
        return 0
    reference = [reference.split()]
    hypothesis = hypothesis.split()
    smoothing_function = SmoothingFunction().method1
    return sentence_bleu(reference, hypothesis, smoothing_function=smoothing_function)

def calculate_cer(reference, hypothesis):
    if pd.isna(hypothesis):
        return 1
    return cer(reference, hypothesis)

def calculate_insertion_ratio(reference, hypothesis):
    if pd.isna(hypothesis):
        return 0
    measures = compute_measures(reference, hypothesis)
    insertions = measures['insertions']
    total_words = len(reference.split())
    return insertions / total_words if total_words > 0 else 0

def calculate_rouge(reference, hypothesis):
    if pd.isna(hypothesis):
        return 0, 0, 0
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return scores['rouge1'].fmeasure, scores['rouge2'].fmeasure, scores['rougeL'].fmeasure

def calculate_wil(reference, hypothesis):
    if pd.isna(hypothesis):
        return 1
    return wil(reference, hypothesis)

def calculate_ril(reference, hypothesis):
    if pd.isna(hypothesis):
        return 1 
    wer_value = compute_measures(reference, hypothesis)['wer']
    length_hypothesis = len(hypothesis.split())
    length_reference = len(reference.split())
    return (wer_value * length_hypothesis) / length_reference

def calculate_meteor(reference, hypothesis):
    if pd.isna(hypothesis):
        return 0
    return meteor_score([reference.split()], hypothesis.split())

metrics = ['BLEU', 'CER', 'Insertion', 'ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'WIL', 'RIL', 'METEOR']

for service in services:
    groundtruth_col = f'groundtruth_{version}'
    column_name = f'{service}_{version}'
    
    df[f'{column_name}_BLEU'] = df.apply(lambda row: calculate_bleu(row[groundtruth_col], row[column_name]), axis=1)
    df[f'{column_name}_CER'] = df.apply(lambda row: calculate_cer(row[groundtruth_col], row[column_name]), axis=1)
    df[f'{column_name}_Insertion'] = df.apply(lambda row: calculate_insertion_ratio(row[groundtruth_col], row[column_name]), axis=1)
    df[[f'{column_name}_ROUGE-1', f'{column_name}_ROUGE-2', f'{column_name}_ROUGE-L']] = df.apply(
        lambda row: pd.Series(calculate_rouge(row[groundtruth_col], row[column_name])), axis=1)
    df[f'{column_name}_WIL'] = df.apply(lambda row: calculate_wil(row[groundtruth_col], row[column_name]), axis=1)
    df[f'{column_name}_RIL'] = df.apply(lambda row: calculate_ril(row[groundtruth_col], row[column_name]), axis=1)
    df[f'{column_name}_METEOR'] = df.apply(lambda row: calculate_meteor(row[groundtruth_col], row[column_name]), axis=1)

higher_better = ['BLEU', 'ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'METEOR']
lower_better = ['CER', 'Insertion', 'WIL', 'RIL']

thresholds = {}
whisper_metrics = [f'Whisper_{version}_{metric}' for metric in metrics]

for column in whisper_metrics:
    if any(metric in column for metric in higher_better):
        threshold = np.percentile(df[column].dropna(), 10)
        thresholds[column] = threshold
        print(f"{column}: 10th percentile threshold = {threshold}")
    elif any(metric in column for metric in lower_better):
        threshold = np.percentile(df[column].dropna(), 90)
        thresholds[column] = threshold
        print(f"{column}: 90th percentile threshold = {threshold}")

conditions = []
for column, threshold in thresholds.items():
    if any(metric in column for metric in higher_better):
        conditions.append(df[column] < threshold)
    elif any(metric in column for metric in lower_better):
        conditions.append(df[column] > threshold)

combined_condition = pd.concat(conditions, axis=1).any(axis=1)

hallucination_candidates = df[combined_condition]
final_df = hallucination_candidates[['segment_name', 'Group', 'groundtruth', 'Whisper_orig']]

final_df.to_csv('hallucination/hallucination_check_category.csv', index=False)

summary_df = pd.DataFrame(columns=['Service', 'Group', 'Version'] + metrics)

for service in services:
    for group in ['aphasia', 'control']:
        group_df = df[df['Group'] == group]
        row_data = {'Service': service, 'Group': group, 'Version': version}
        
        for metric in metrics:
            column_name = f'{service}_{version}_{metric}'
            row_data[metric] = f"{group_df[column_name].mean():.4f}"
            
        summary_df = pd.concat([summary_df, pd.DataFrame([row_data])], ignore_index=True)

def generate_latex_table_with_subcategories(df, services, metrics):
    latex_table = r"""\begin{table}[ht]
\centering
\caption{Suite of average automated ASR metrics (higher is better for BLEU, ROUGE-1, ROUGE-2, ROUGE-L, and METEOR; lower is better for CER, Insertion, WIL, and RIL). For each metric, the best performing service is marked in bold. These metrics were used to determine the subset of 815 audio files on which manual review was performed to check for hallucinations.}
\resizebox{\columnwidth}{!}{
\begin{tabular}{l""" + "c" * len(metrics) + r"""}
\toprule
\textbf{ASR Service} & """ + " & ".join([f"\\textbf{{{metric}}}" for metric in metrics]) + r""" \\
\midrule"""

    best_values = {}
    for group in ['aphasia', 'control']:
        best_values[group] = {}
        for metric in metrics:
            group_data = df[(df['Group'] == group) & (df['Version'] == version)]
            if metric in higher_better:
                best_service = group_data.loc[group_data[metric].astype(float).idxmax()]['Service']
            else:  # lower is better
                best_service = group_data.loc[group_data[metric].astype(float).idxmin()]['Service']
            best_values[group][metric] = best_service

    for service in services:
        latex_table += f"\n{service}".ljust(15) + r" & & & & & & & & & \\"
        
        for group in ['aphasia', 'control']:
            group_name = "\\quad Aphasia" if group == 'aphasia' else "\\quad Control"
            group_rows = df[(df['Service'] == service) & (df['Group'] == group) & (df['Version'] == version)]
            
            if not group_rows.empty:
                latex_table += f"\n{group_name}".ljust(15) + " & "
                
                metric_values = []
                for metric in metrics:
                    value = group_rows.iloc[0][metric]
                    if best_values[group][metric] == service:
                        metric_values.append(f"\\textbf{{{value}}}")
                    else:
                        metric_values.append(value)
                        
                latex_table += " & ".join(metric_values)
                latex_table += r" \\"
                
    latex_table += r"""
\bottomrule
\end{tabular}
}
\label{tab:asr_metrics}
\end{table}"""
    
    return latex_table

latex_table = generate_latex_table_with_subcategories(summary_df, services, metrics)
with open("ASR_Services_Metrics_Table.tex", "w") as f:
    f.write(latex_table)

print("Processing complete. Generated 'hallucination_candidates.csv' and 'ASR_Services_Metrics_Table.tex'")