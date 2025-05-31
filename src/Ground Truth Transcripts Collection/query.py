import os
import pandas as pd
import re
from datetime import datetime

current_date = datetime.now().strftime('%Y-%m-%d')
print(current_date)

# drop rows with 'UNK'
aphasia_all = pd.read_csv(f"../../data/aphasia_all_fix_{current_date}.csv")
control_all = pd.read_csv(f"../../data/control_all_fix_{current_date}.csv")

aphasia_all['clean_v2'] = aphasia_all['clean_v2'].fillna('')
control_all['clean_v2'] = control_all['clean_v2'].fillna('')

# standardize whitespace
aphasia_all['clean_v2'] = aphasia_all['clean_v2'].apply(lambda text: re.sub("\s+", " ", text.strip()))
control_all['clean_v2'] = control_all['clean_v2'].apply(lambda text: re.sub("\s+", " ", text.strip()))

aphasia_all = aphasia_all[~aphasia_all['clean_v2'].str.contains('UNK')]
control_all = control_all[~control_all['clean_v2'].str.contains('UNK')]

aphasia_all.to_csv(f"../../data/aphasia_nounk_{current_date}.csv", index=False)
control_all.to_csv(f"../../data/control_nounk_{current_date}.csv", index=False)

aphasia_all['clean_v2'] = aphasia_all['clean_v2'].fillna('')
control_all['clean_v2'] = control_all['clean_v2'].fillna('')

aphasia_all['word_count_v2'] = aphasia_all['clean_v2'].apply(lambda x: len(x.split()))
control_all['word_count_v2'] = control_all['clean_v2'].apply(lambda x: len(x.split()))

aphasia_filtered = aphasia_all[aphasia_all['word_count_v2'] > 3]
control_filtered = control_all[control_all['word_count_v2'] > 3]
# filter out MMA20a.cha
aphasia_filtered = aphasia_filtered[aphasia_filtered['filename'] != 'MMA20a.cha']

aphasia_filtered['duration'] = aphasia_filtered['end_time'] - aphasia_filtered['start_time']
control_filtered['duration'] = control_filtered['end_time'] - control_filtered['start_time']

aphasia_filtered.to_csv(f'../../data/aphasia_nounk_over3_{current_date}.csv', index=False)
control_filtered.to_csv(f'../../data/control_nounk_over3_{current_date}.csv', index=False)

aphasia_filtered = aphasia_filtered[aphasia_filtered['duration'] >= 1000]
control_filtered = control_filtered[control_filtered['duration'] >= 1000]

aphasia_filtered.to_csv(f"../../data/aphasia_nounk_over3_dur1_{current_date}.csv", index=False)
control_filtered.to_csv(f"../../data/control_nounk_over3_dur1_{current_date}.csv", index=False)