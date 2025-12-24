import os
import argparse
import subprocess
from pathlib import Path
from typing import List, Optional

CSV_COMMENT_CHAR = '#'

def read_csv(path: Path, expected_columns: Optional[List[str]] = None) -> List[List[str]]:
    if not path.exists():
        raise IOError(f'CSV file does not exsit: {path}')

    data = []
    check_header = expected_columns is not None
    with open(str(path), 'r') as fid:
        for line in fid:
            line = line.strip()
            if len(line) == 0:
                continue
            if line[0] == CSV_COMMENT_CHAR:
                if check_header and len(data) == 0:
                    columns = [w.strip() for w in line[1:].split(',')]
                    if columns != expected_columns:
                        raise ValueError(
                            f'Got CSV columns {columns} but expected {expected_columns}.')
                check_header = False
            else:
                words = [w.strip() for w in line.split(',')]
                data.append(words)
    return data


def extract_frames_from_video(input_dir: Path, images_dir: Path):
    assert images_dir.exists()
    video_path = input_dir / 'images.mp4'

    # Extract frames.
    frames_format = 'out-%012d.jpg'
    cmd = [
        'ffmpeg',
        '-hide_banner', '-loglevel', 'warning', '-nostats',
        '-i', video_path.as_posix(),
        '-fps_mode', "passthrough",
        # '-vsync', '0',
        '-qmin', '1',
        '-q:v', '1',
        (images_dir / frames_format).as_posix(),
    ]
    subprocess.run(cmd, check=True)
    print("ffmpeg -> Done")

    # Extract timestamps.
    cmd = [
        'ffprobe',
        '-hide_banner', '-loglevel', 'warning',
        '-f', 'lavfi',
        '-i', f'movie={video_path.as_posix()}',
        # '-show_entries', 'frame=pkt_pts',
        # '-select_streams', 'v:0',
        '-show_entries', 'frame=pts_time',
        # '-show_entries', 'frame=best_effort_timestamp_time',
        '-of', 'csv=p=0',
        # video_path.as_posix()
    ]
    result = subprocess.run(cmd,
                            check=True,
                            capture_output=True,
                            text=True)
    print("ffprobe -> Done")
    
    # Convert list of newline separated chars to list of strings.
    # timestamps = ''.join(result.stdout).split()
    timestamps = [
        int(round(float(x.rstrip(',')) * 1e6))
        for x in result.stdout.splitlines()
    ]

    # print(timestamps)
    # Extract time origin (timestamp of the first pose).
    poses = read_csv(input_dir / 'poses.txt')
    # print(len(poses), len(timestamps))
    assert len(poses) == len(timestamps)
    time_origin = int(poses[0][0])
    # print(time_origin)

    # sys.exit(0)
    
    # Rename all image data.
    for idx, timestamp in enumerate(timestamps):
        image_path = images_dir / (frames_format % (idx + 1))
        # output_path = images_dir / (str(time_origin + int(timestamp)) + '.jpg')
        output_path = images_dir / (str(time_origin + int(timestamp)) + '.jpg')
        image_path.rename(output_path)
    
    print("DONE!")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    # parser.add_argument("images_dir", required=True)
    args = parser.parse_args().__dict__
    
    input_dir = Path(args["input_dir"])
    # images_dir = Path(args["images_dir"])
    images_dir = input_dir / "frames"
    
    if not images_dir.exists():
        images_dir.mkdir(parents=True, exist_ok=True)
    
    extract_frames_from_video(input_dir, images_dir)