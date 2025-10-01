import os
import math
import shutil
from datetime import datetime, timedelta


def parse_time(name):
    if name.startswith("hl_"):
        base, micros = name.split('.')
        dt = datetime.strptime(base.replace("hl_", ""), "%Y-%m-%d-%H-%M-%S-%f")
        return dt + timedelta(microseconds=int(micros) * 1000)
    elif name.startswith("spot_"):
        return datetime.strptime(name.replace("spot_", ""), "%Y-%m-%d-%H-%M")
    elif name.startswith("ios_"):
        return datetime.strptime(name.replace("ios_", ""), "%Y-%m-%d_%H.%M.%S_%f")
    else:
        raise ValueError(f"Unknown format: {name}")

def extract_subsessions(session, endpoint_dir, map_subsessions, expand_minutes=5):
    subs_path = os.path.join(session, "proc/subsessions.txt")
    save_path = os.path.join(endpoint_dir, "proc/subsessions.txt")

    map_times = [parse_time(name) for name in map_subsessions]
    # time_min = min(map_times) - timedelta(minutes=expand_minutes)
    time_max = max(map_times) + timedelta(minutes=expand_minutes)

    with open(subs_path, 'r', encoding='utf-8') as f:
        subs = [line.strip() for line in f if line.strip()]

    # result = [s for s in subs if time_min <= parse_time(s) <= time_max]
    result = [s for s in subs if parse_time(s) <= time_max]
    if len(result) == 0:
        result = subs[:1]

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        f.writelines(s + '\n' for s in result)

    return result

def extract_proc(session, endpoint_dir, map_subsessions, expand_minutes=5):
    subsessions = extract_subsessions(session, endpoint_dir, map_subsessions, expand_minutes=expand_minutes)
    return subsessions

def extract_raw_data(session, endpoint_dir, subsessions, symlinks=True):
    raw_data_dir = os.path.join(session, "raw_data")
    save_dir = os.path.join(endpoint_dir, "raw_data")

    os.makedirs(save_dir, exist_ok=True)

    raw_data_sessions = os.listdir(raw_data_dir)
    for raw_data_session in raw_data_sessions:
        if raw_data_session in subsessions:
            src_path = os.path.join(raw_data_dir, raw_data_session)
            dst_path = os.path.join(save_dir, raw_data_session)

            shutil.copytree(src_path, dst_path, symlinks=symlinks, dirs_exist_ok=True)
            
def extract_file(query_dir, endpoint_dir, file_name, subsessions, key_index):
    file_path = os.path.join(query_dir, file_name)
    save_path = os.path.join(endpoint_dir, file_name)
    
    if not os.path.exists(file_path):
        return
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        for line in lines:
            if line.startswith('#'):
                f.write(line)
                continue
            key = line.split(', ')[key_index].split('/')[0]
            if key not in subsessions:
                continue
            f.write(line)


def extract_query(query_dir, endpoint_dir, map_subsessions, expand_minutes=5,  symlinks=True):
    device = query_dir.split('/')[-1].split('_query')[0]
    
    subsessions = extract_proc(query_dir, endpoint_dir, map_subsessions, expand_minutes)
    extract_raw_data(query_dir, endpoint_dir, subsessions, symlinks=symlinks)
    extract_file(query_dir, endpoint_dir, 'proc/keyframes_pruned_subsampled.txt', subsessions, 1)
    extract_file(query_dir, endpoint_dir, 'images.txt', subsessions, 1)
    extract_file(query_dir, endpoint_dir, 'sensors.txt', subsessions, 0)
    
    if device in ['hl', 'spot']:
        extract_file(query_dir, endpoint_dir, 'rigs.txt', subsessions, 1)
    print('DONE', query_dir)