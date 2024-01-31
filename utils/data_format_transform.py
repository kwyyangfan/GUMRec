import json
import os
import argparse
import pandas as pd

def filter_and_convert(input_file, target_id):
    filtered_data = []

    output_file_name = os.path.splitext(input_file)[0] 
    if target_id is not None:
        output_file_name += '_' + str(target_id) 
    else:
        output_file_name += '_all' 

    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            if target_id is None or target_id=='all' or data['id'] == target_id:
                filtered_data.append({
                    'instruction': data['prompt'],
                    'input': '',
                    'output': data['completion']
                })

    output_file = output_file_name + '.json'

    with open(output_file, 'w', encoding='utf-8') as f:
        for data in filtered_data:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')

    print(f"The reformatted and filtered document has been saved as：{output_file}")
def filter_and_convert_rec(input_file, target_id):
    filtered_data = []

    output_file_name = os.path.splitext(input_file)[0] 
    if target_id is not None:
        output_file_name += '_' + str(target_id)  
    else:
        output_file_name += '_all'  

    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            if target_id is None or target_id=='all' or data['task_type'] == target_id:
                filtered_data.append({
                    'instruction': data['source'],
                    'input': '',
                    'output': data['target']
                })

    output_file = output_file_name + '.json'

    with open(output_file, 'w', encoding='utf-8') as f:
        for data in filtered_data:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')

    print(f"The reformatted and filtered document has been saved as：{output_file}")
def convert_rec(input_file):
    filtered_data = []

    output_file_name = os.path.splitext(input_file)[0] 


    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            if target_id is None or target_id=='all' or data['task_type'] == target_id:
                filtered_data.append({
                    'instruction': data['source'],
                    'input': '',
                    'output': data['target']
                })

    output_file = output_file_name + '.json'

    with open(output_file, 'w', encoding='utf-8') as f:
        for data in filtered_data:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')

    print(f"The reformatted and filtered document has been saved as：{output_file}")
def word_count(input_file):
    max_word_count=0
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            source_sentence = data['source']
            word_count = len(source_sentence.split()) 
            max_word_count = max(max_word_count, word_count) 
    print(f"The maximum number of words in a sentence in the file is: {max_word_count}")

def split_csv_file(file_path, num_parts):
    df = pd.read_csv(file_path)

    total_rows = len(df)
    rows_per_part = total_rows // num_parts

    file_name = os.path.basename(file_path)
    file_name_without_ext = os.path.splitext(file_name)[0]
    file_directory = os.path.dirname(file_path)

    for i in range(num_parts):
        start_index = i * rows_per_part
        end_index = start_index + rows_per_part

        if i == num_parts - 1:
            end_index = total_rows

        part_df = df[start_index:end_index]
        part_file_name = f"{file_name_without_ext}_{i + 1}.csv"
        part_file_path = os.path.join(file_directory, part_file_name)
        part_df.to_csv(part_file_path, index=False)


def merge_csv_files(file_name, output_file):
    file_directory = os.path.dirname(output_file)
    csv_files = [file for file in os.listdir(file_directory) if file.startswith(file_name) and file.endswith('.csv') and file!=file_name+'.csv']
    csv_files.sort() 

    merged_df = pd.concat([pd.read_csv(os.path.join(file_directory, file)) for file in csv_files])

    merged_df.to_csv(output_file, index=False)

def compare_csv_files(file1, file2):
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    if df1.equals(df2):
        print("Two CSV files are identical")
    else:
        print("Two CSV files are not exactly the same")

if __name__ == '__main__':
    word_count('data/chatglm-data-full/full_train.json')