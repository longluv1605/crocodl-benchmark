import os
from .extract_map import extract_map
from .extract_query import extract_query


def group(sessions):
    devices = {}
    for session in sessions:
        device, type = session.split('_')
        if not device in devices:
            devices[device] = {}
        devices[device][type] = session
    return devices

def extract_location(location_dir, endpoint_dir, ext_percent=0.3, symlinks=True):
    location_dir = os.path.join(location_dir, "sessions")
    endpoint_dir = os.path.join(endpoint_dir, "sessions")
    
    sessions = sorted(os.listdir(location_dir))
    devices = group(sessions)
    for device in devices.values():
        map, query = device.values()
        
        map_dir = os.path.join(location_dir, map)
        map_endpoint_dir = os.path.join(endpoint_dir, map)
        query_dir = os.path.join(location_dir, query)
        query_endpoint_dir = os.path.join(endpoint_dir, query)
        
        map_subsessions = extract_map(map_dir, map_endpoint_dir, ext_percent=ext_percent,  symlinks=symlinks)
        extract_query(query_dir, query_endpoint_dir, map_subsessions, expand_minutes=5,  symlinks=True)