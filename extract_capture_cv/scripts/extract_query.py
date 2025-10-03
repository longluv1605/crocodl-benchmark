import os
import math
import shutil
import pandas as pd
import numpy as np


def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

def write_file(file_path, lines):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as  f:
        for line in lines:
            f.write(line + '\n')

def extract_proc(map_endpoint_dir, query_endpoint_dir, file_name="proc/subsessions.txt"):
    src_path = os.path.join(map_endpoint_dir, file_name)
    des_path = os.path.join(query_endpoint_dir, file_name)
    
    os.makedirs(os.path.dirname(des_path), exist_ok=True)
    shutil.copyfile(src_path, des_path)

def get_query_info(map_endpoint_dir, file_name='images.txt', ext_percent=0.3):
    file_path = os.path.join(map_endpoint_dir, file_name)
    df = pd.read_csv(file_path, sep=", ", engine="python")
    all_ids = df['# timestamp'].unique()

    query_len = math.ceil(len(all_ids) * ext_percent)
    query_ids = np.random.choice(all_ids, query_len, replace=False)
    
    # Get all image paths for the selected query IDs
    query_df = df[df['# timestamp'].isin(query_ids)]
    query_image_paths = query_df['image_path'].values
    
    return query_ids, query_image_paths

def line_tag(line, query_ids, key_index, splitter="_"):
    idx = line.split(", ")[0].split("/")[-1].split(splitter)[key_index]
    if idx.isnumeric() and int(idx) in query_ids:
        return 'query'
    if not idx.isnumeric():
        return 'all'
    return 'map'

def split_file(map_endpoint_dir, query_endpoint_dir, query_ids, file_name, key_index, splitter="_"):
    src_path = os.path.join(map_endpoint_dir, file_name)
    des_path = os.path.join(query_endpoint_dir, file_name)
    
    map_lines, query_lines = [], []
    all_lines =  read_file(src_path)
    for line in all_lines:
        match line_tag(line, query_ids, key_index, splitter):
            case "query":
                query_lines.append(line)
            case "map":
                map_lines.append(line)
            case "all":
                query_lines.append(line)
                map_lines.append(line)
    
    write_file(src_path, map_lines)            
    write_file(des_path, query_lines)
    
def split_txt(map_endpoint_dir, query_endpoint_dir, query_ids, device):
    
    key_index = -1
    splitter = "_"
    split_file(map_endpoint_dir, query_endpoint_dir, query_ids, "images.txt", key_index=key_index, splitter=splitter)
    split_file(map_endpoint_dir, query_endpoint_dir, query_ids, "sensors.txt", key_index=key_index, splitter=splitter)
    split_file(map_endpoint_dir, query_endpoint_dir, query_ids, "trajectories.txt", key_index=key_index, splitter=splitter)
    
    if device in ['hl', 'spot']:
        if device == 'spot':
            key_index = 0
            splitter = "-"
        split_file(map_endpoint_dir, query_endpoint_dir, query_ids, "rigs.txt", key_index=key_index, splitter=splitter)          

def split_raw(map_endpoint_dir, query_image_paths, raw_dir="raw_data"):
    map_raw_dir = os.path.join(map_endpoint_dir, raw_dir)
    
    # Process each query image path directly
    for relative_image_path in query_image_paths:
        # Build full source path
        source_path = os.path.join(map_raw_dir, relative_image_path)
        
        # Check if the source file exists
        if os.path.exists(source_path):
            # Create destination path by replacing "_map" with "_query" in the full path
            dest_path = source_path.replace("_map", "_query")
            
            # Create destination directory
            dest_dir = os.path.dirname(dest_path)
            os.makedirs(dest_dir, exist_ok=True)
            
            # Move the file (remove from map, add to query)
            shutil.move(source_path, dest_path)

def create_keyframes(query_endpoint_dir, device):
    save_path = os.path.join(query_endpoint_dir, "proc", "keyframes_original.txt")

    if device == 'ios':
        file_name = "images.txt"
        cols = ["# timestamp", "sensor_id"]
    else:
        file_name = "rigs.txt"
        cols = ["# timestamp", "# rig_id"]
        
        id_index = 0 if device == 'spot' else -1
        spt = "-" if device == 'spot' else "_"
        
        def f(x):
            return x.split("/")[-1].split(spt)[id_index]
        
    ref_path = os.path.join(query_endpoint_dir, file_name)
    ref = pd.read_csv(ref_path, sep=", ", engine='python')
    
    if device != 'ios':
        ref["# timestamp"] = ref["# rig_id"].apply(lambda x: f(x))
        
    res = ref[cols].drop_duplicates().copy()
    with open(save_path, "w") as f:
        for row in res.values:
            f.write(", ".join(map(str, row)) + "\n")

def extract_query(map_endpoint_dir, query_endpoint_dir, ext_percent=0.3):
    device = query_endpoint_dir.split('/')[-1].split('_query')[0]
    '''
    SAME: 
        proc/subsessions.txt
    SPLIT: by id (timestamps)
        sensors.txt (only ios is diff)
        images.txt
        trajectories.txt
        rigs.txt (for hl and spot)
    RAW: move by id
    '''
    # PROC
    extract_proc(map_endpoint_dir,  query_endpoint_dir)
    
    query_ids, query_image_paths = get_query_info(map_endpoint_dir, 'images.txt', ext_percent)
    print(f"--> {device}_query len = {len(query_ids)}")
    
    # .TXT
    split_txt(map_endpoint_dir, query_endpoint_dir, query_ids, device)
    
    # RAW
    split_raw(map_endpoint_dir, query_image_paths)

    # keyframes
    create_keyframes(query_endpoint_dir, device)
    
    print("DONE", query_endpoint_dir)