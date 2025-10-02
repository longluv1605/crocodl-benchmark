import os
import math
import shutil


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

def extract_subsessions(map_dir, endpoint_dir, ext_percent=0.3):
    subsessions_path = os.path.join(map_dir, "proc/subsessions.txt")
    save_path = os.path.join(endpoint_dir, "proc/subsessions.txt")
    
    subsessions = read_file(subsessions_path)
    
    pos = math.ceil(len(subsessions) * ext_percent)
    subsessions = subsessions[:pos]
    
    write_file(save_path, subsessions)
            
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
            
def extract_file(map_dir, endpoint_dir, file_name, subsessions, key_index):
    file_path = os.path.join(map_dir, file_name)
    save_path = os.path.join(endpoint_dir, file_name)
    
    if not os.path.exists(file_path):
        print(file_path)
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


def extract_map(map_dir, endpoint_dir, ext_percent=0.3,  symlinks=True):
    device = map_dir.split('/')[-1].split('_map')[0]
    
    subsessions = extract_subsessions(map_dir, endpoint_dir, ext_percent)
    extract_raw_data(map_dir, endpoint_dir, subsessions, symlinks=symlinks)
    extract_file(map_dir, endpoint_dir, 'images.txt', subsessions, 1)
    extract_file(map_dir, endpoint_dir, 'sensors.txt', subsessions, 0)
    extract_file(map_dir, endpoint_dir, 'trajectories.txt', subsessions, 1)
    
    if device in ['hl', 'spot']:
        extract_file(map_dir, endpoint_dir, 'rigs.txt', subsessions, 1)
    print('DONE', endpoint_dir)
    