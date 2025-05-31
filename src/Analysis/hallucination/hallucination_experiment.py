# Fig S7 generation

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.contingency_tables import mcnemar
from statannot import add_stat_annotation

df = pd.read_csv('experiment_wer_rates_2024-07-03.csv')
experiment_order = ['original', 'silent_beginning', 'whitenoise_beginning', 'whitenoise_insertion', 'whitenoise_throughout', 
                    'realnoise_throughout_lowSNR', 'realnoise_throughout_highSNR', 'cut_audio']
experiment_rename = {
    'original': 'Original',
    'silent_beginning': 'Silence, \nbeginning',
    'whitenoise_beginning': 'White noise, \nbeginning',
    'whitenoise_insertion': 'White noise, \nmiddle',
    'whitenoise_throughout': 'White noise, \nthroughout',
    'realnoise_throughout_lowSNR': 'Real-life noise \nlow SNR, throughout',
    'realnoise_throughout_highSNR': 'Real-life noise \nhigh SNR, throughout',
    'cut_audio': 'Cut, \nmiddle'
}

def plot_subset_with_pvalues(df_subset, title):
    df_subset['Whisper_hallucination'] = df_subset['Whisper_hallucination'].apply(lambda x: 1 if x == 0.5 else x)
    df_subset['Whisper_hallucination_count'] = df_subset['Whisper_hallucination'].apply(lambda x: 1 if x >= 1 else 0)

    df_whisper = df_subset[['experiment', 'Group', 'Whisper_hallucination_count']].copy()
    df_whisper.rename(columns={'Whisper_hallucination_count': 'Hallucination_Count'}, inplace=True)

    df_whisper['experiment'] = df_whisper['experiment'].replace(experiment_rename)
    df_whisper['experiment'] = pd.Categorical(df_whisper['experiment'], categories=[experiment_rename[exp] for exp in experiment_order], ordered=True)

    relevant_data = df_subset[['experiment', 'Whisper_hallucination', 'original_segment_name']]
    experiment_names = relevant_data['experiment'].unique()
    experiment_names = [exp for exp in experiment_names if exp != 'original']
    
    filtered_data = df_subset[df_subset['experiment'].isin(['original'] + experiment_names)]
    pivot_data = filtered_data.pivot(index='original_segment_name', columns='experiment', values='Whisper_hallucination')
    
    mcnemar_results = {}
    
    for exp in experiment_names:
        if exp in pivot_data.columns:
            comparison_data = pivot_data[['original', exp]].dropna()
            table = pd.crosstab(comparison_data['original'], comparison_data[exp])
            if table.shape == (2, 2):
                result = mcnemar(table, exact=False)  
                mcnemar_results[exp] = {'statistic': result.statistic, 'p-value': result.pvalue}
            else:
                mcnemar_results[exp] = {'statistic': None, 'p-value': None, 'note': 'Insufficient data for test'}
        else:
            mcnemar_results[exp] = {'statistic': None, 'p-value': None, 'note': 'Experiment data not found'}

    plt.figure(figsize=(16, 8))
    barplot = sns.barplot(data=df_whisper, x='experiment', y='Hallucination_Count', hue='Group', estimator=sum, palette="Set1", dodge=True, errorbar=None)
    plt.title(f'Hallucination Counts - {title}')
    plt.ylabel('Count of Hallucinations')
    plt.xlabel('Experiment Types')
    barplot.set_xticklabels([label.get_text().split('_')[0] for label in barplot.get_xticklabels()], rotation=45, ha='right')
    plt.legend(title='Group')

    box_pairs = [(("Original", f"{experiment_rename[exp]}"),) for exp in experiment_names]
    significant_pairs = []
    significant_pvalues = []
    for pair in box_pairs:
        exp_name = pair[0][1].strip()
        for key, val in experiment_rename.items():
            if val == exp_name:
                original_name = key
                break
        if original_name in mcnemar_results and mcnemar_results[original_name]['p-value'] is not None and mcnemar_results[original_name]['p-value'] < 0.05:
            significant_pairs.append(pair[0])
            significant_pvalues.append(mcnemar_results[original_name]['p-value'])

    add_stat_annotation(barplot, data=df_whisper, x='experiment', y='Hallucination_Count', 
                        box_pairs=significant_pairs, perform_stat_test=False, pvalues=significant_pvalues, 
                        test_short_name='McNemar', text_format='star', loc='inside', verbose=2, line_offset_to_box=0.9)

    handles, labels = barplot.get_legend_handles_labels()
    star_labels = ['* (p<0.05)', '** (p<0.01)', '*** (p<0.001)', '**** (p<0.0001)']
    for star_label in star_labels:
        handles.append(plt.Line2D([0], [0], color='black', marker='*', linestyle='None'))
        labels.append(star_label)
    barplot.legend(handles=handles, labels=labels)
    plt.savefig('../figures/hallucination_count.pdf', format='pdf', bbox_inches='tight')
    plt.show()

plot_subset_with_pvalues(df, "Whisper")