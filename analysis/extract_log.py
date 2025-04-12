import re
import json
import pandas as pd
import os
def log_to_xlsx(log_path):
    log_filename = os.path.splitext(os.path.basename(log_path))[0]
    output_file = f"{log_filename}.xlsx"
    log_pattern = re.compile(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - INFO - ({.*})')
    fold_data = {i: {'labels': [], 'preds': [], 'probs': []} for i in range(1, 6)}
    with open(log_path, 'r', encoding='utf-8') as file:
        for line in file:
            match = log_pattern.search(line)
            if match:
                log_time, json_str = match.groups()
                data = json.loads(json_str)
                fold = data.get('fold')
                labels = data.get('labels')
                preds = data.get('preds')
                probs = data.get('probs')

                if fold and labels and preds and probs:
                    fold_data[fold]['labels'].extend(labels)
                    fold_data[fold]['preds'].extend(preds)
                    fold_data[fold]['probs'].extend(probs)
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        for fold, data in fold_data.items():
            if data['labels']:
                max_length = max(len(data['labels']), len(data['preds']), len(data['probs']))
                df = pd.DataFrame({
                    'Labels': data['labels'] + [None] * (max_length - len(data['labels'])),
                    'Preds': data['preds'] + [None] * (max_length - len(data['preds'])),
                    'Probs': data['probs'] + [None] * (max_length - len(data['probs']))
                })
                probs_df = pd.DataFrame(df['Probs'].tolist(), columns=['class0', 'class1'])
                df = pd.concat([df.drop('Probs', axis=1), probs_df], axis=1)
                df.to_excel(writer, sheet_name=f'Fold_{fold}', index=False)

    return output_file

if __name__ == "__main__":
    test_log = r"U:\\work2-dti\\logs\\NCvsPD\\contrastive_model2_20250406_232130_38.89.log"
    output_file = log_to_xlsx(test_log)
    print(f"数据已成功写入 {output_file}")
