import cv2
import json
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from tqdm.contrib.concurrent import thread_map
from collections import defaultdict

from .pair_selection import PairSelection
from .feature_extraction import FeatureExtraction
from .feature_matching import FeatureMatching
from ..utils.misc import same_configs, write_config
from ..utils.cross_validate import estimate_pose, compute_pose_error
from ..utils.capture import list_images_for_session, list_trajectory_keys_for_session
from scantools.capture import Capture, Trajectories, Pose
from scantools.proc.alignment.image_matching import get_keypoints, get_matches

from typing import Tuple

import logging
logger = logging.getLogger(__name__)



class CrossValidationPaths:
    def __init__(self, root, config, query_id, ref_id):
        self.root = root
        self.workdir = (
            root / 'cross_validation_1' / query_id / ref_id
            / config['features']['name'] / config['matches']['name']
            / config['pairs']['name'] / config['name']
        )
        self.poses = self.workdir / 'poses.txt'
        self.config = self.workdir / 'configuration.json'
        self.evaluation = self.workdir / 'evaluation.json'
        self.detailed_errors = self.workdir / "detailed_errors.txt"
        self.fail_moderate = self.workdir / "fail_moderate.txt"
        self.fail_severe = self.workdir / "fail_severe.txt"
        

class CrossValidation:
    methods = {}
    method2class = {}
    method = None
    Rt_threshold = (20.0, 10.0)
    top = 3
    
    def __init_subclass__(cls):
        '''Register the child classes into the parent'''
        if cls.method is None:  # abstract class
            return
        name = cls.method['name']
        cls.methods[name] = cls.method
        cls.method2class[name] = cls

    def __new__(cls, config, *_, **__):
        '''Instanciate the object from the child class'''
        return super().__new__(cls.method2class[config['name']])

    
    def __init__(
        self,
        config: dict,
        outputs,
        capture: Capture,
        query_id,
        ref_id,
        pair_selection: PairSelection,
        extraction: FeatureExtraction,
        extraction_ref: FeatureExtraction,
        matching: FeatureMatching,
        query_keys: list = None,
        Rt_threshold: Tuple[float, float] = (20.0, 20.0),
        filter_R_threshold: Tuple[float, float] = (5.0, 15.0),
        top: int = 3,
        parallel: bool = True,
    ):
        self.parallel = parallel
        self.Rt_threshold = Rt_threshold
        self.filter_R_threshold = filter_R_threshold
        self.top = top
        
        assert query_id == extraction.session_id
        assert query_id == matching.query_id
        assert ref_id == extraction_ref.session_id
        assert ref_id == matching.ref_id
        
        self.config = config = {
            **deepcopy(config),
            'features': extraction.config,
            'matches': matching.config,
            'pairs': matching.pair_selection.config.to_dict(),
        }
        
        self.query_id = query_id
        self.ref_id = ref_id
        self.paths = CrossValidationPaths(outputs, self.config, query_id, ref_id)
        self.query_keys = query_keys
        
        # Primary variables
        self.pairs = pair_selection.pairs # [[query_img, map_img], ...]
        self.extraction = extraction
        self.extraction_ref = extraction_ref
        self.matching = matching
        self.session_q = capture.sessions[query_id] # (sensors, rigs, trajectories)
        self.session_r = capture.sessions[ref_id] # (sensors, rigs, trajectories)

        self.paths.workdir.mkdir(parents=True, exist_ok=True)
        overwrite = not same_configs(self.config, self.paths.config)
        if overwrite:
            logger.info('Cross validating session %s with features %s.',
                        query_id, self.config['features']['name'])
            self.poses = self.estimate_poses(capture)
            self.poses.save(self.paths.poses)
        else:
            self.poses = Trajectories().load(self.paths.poses)
            
        self.err_t, self.err_R, self.recall, self.detailed_errors, self.moderate_fails, self.severe_fails = self.compute_errors(capture)
        self.evaluation = {
            query_id: {
                ref_id: {
                    'err_t': self.err_t,
                    'err_R': self.err_R,
                    'acc_R': 1 - self.err_R / 90.0,
                    'recall': self.recall
                }
            }
        }
        write_config(self.evaluation, self.paths.evaluation)
        write_config(self.config, self.paths.config)

        with open(self.paths.detailed_errors, "w") as f:
            for item in self.detailed_errors:
                f.write(f"{item['query']} {item['map']} err_R={item['err_R']:.3f} err_t={item['err_t']:.3f}\n")

        with open(self.paths.fail_moderate, "w") as f:
            for item in self.moderate_fails:
                f.write(f"{item['query']} {item['map']} err_R={item['err_R']:.2f} err_t={item['err_t']:.2f}\n")
        
        with open(self.paths.fail_severe, "w") as f:
            for item in self.severe_fails:
                f.write(f"{item['query']} {item['map']} err_R={item['err_R']:.2f} err_t={item['err_t']:.2f}\n")
        
        print(f"R_threshold = {self.Rt_threshold[0]} || t_threshold = {self.Rt_threshold[1]}")        
        print(f"moderate_R_threshold = {self.filter_R_threshold[0]} || severe_R_threshold = {self.filter_R_threshold[1]}") 
        print(f"Translation err_t {query_id}-{ref_id} = {self.err_t}")
        print(f"Rotation err_R {query_id}-{ref_id} = {self.err_R}")
        print(f"Rotation acc_R {query_id}-{ref_id} = {1 - self.err_R / 90.0}")
        print(f"Recall recall {query_id}-{ref_id} = {self.recall}")
    
    def estimate_poses(self, capture):     
        query_keys, query_names, _ = list_images_for_session(capture, self.query_id, self.query_keys)
        map_keys, map_names, _ = list_images_for_session(capture, self.ref_id)
        query_rigs = list_trajectory_keys_for_session(capture, self.query_id, self.query_keys)
                        
        poses = Trajectories()

        def _worker_fn(idx: int):
            query_img, map_img = self.pairs[idx]
            
            kpts0, _ = get_keypoints(self.extraction.paths.features, [query_img])
            kpts1, _ = get_keypoints(self.extraction_ref.paths.features, [map_img])
            matches = get_matches(self.matching.paths.matches, [(query_img, map_img)])
            
            kpts0 = np.asarray(kpts0[0])
            kpts1 = np.asarray(kpts1[0])
            matches = np.asarray(matches[0])

            mkpts0 = kpts0[matches[:, 0]]
            mkpts1 = kpts1[matches[:, 1]]
            
            q_ts, query_cam = query_keys[query_names.index(query_img)]
            r_ts, map_cam = map_keys[map_names.index(map_img)]
            K0 = self.session_q.sensors[query_cam].K
            K1 = self.session_r.sensors[map_cam].K
            
            pose = estimate_pose(mkpts0, mkpts1, K0, K1, thresh=self.method['thresh'])
            
            device_id = next((rig for ts, rig in query_rigs if ts == q_ts), None)
            if pose is not None:
                key = (f"{q_ts}-{r_ts}", device_id)
                R, T, _ = pose
                pose = Pose(R, T)
                poses[key] = pose

        map_ = thread_map if self.parallel else lambda f, x: list(map(f, tqdm(x)))
        map_(_worker_fn, range(len(self.pairs)))
        return poses

    def filter_top(self, results: dict, n: int = 5):
        for k, v in results.items():
            results[k] = sorted(v)[:n]
        return results

    def convert_to_list(self, results: dict, top: int = 5):
        results = self.filter_top(results, top)
        result_list = []
        for value_list in results.values():
            result_list.extend(value_list)
        return np.array(result_list)

    def compute_errors(self, capture):
        """
        Need: pairs, Q/R images, est poses, Q/R poses, Q/R rigs
        """      
        query_keys, query_names, _ = list_images_for_session(capture, self.query_id, self.query_keys)
        map_keys, map_names, _ = list_images_for_session(capture, self.ref_id)
        query_rigs = list_trajectory_keys_for_session(capture, self.query_id, self.query_keys)
        # map_rigs = list_trajectory_keys_for_session(capture, self.ref_id)
                
        all_err_t, all_err_R = defaultdict(list), defaultdict(list)
        detailed_results = []
        moderate_fails = []
        severe_fails = [] 
        def _worker_fn(idx: int):
            query_img, map_img = self.pairs[idx]
            q_ts, query_sensor_id = query_keys[query_names.index(query_img)]
            r_ts, map_sensor_id = map_keys[map_names.index(map_img)]
            
            device_id = next((rig for ts, rig in query_rigs if ts == q_ts), None)
            key = (f"{q_ts}-{r_ts}", device_id)
            
            if key not in self.poses: return
            est_pose = self.poses[key].to_4x4mat()
            
            query_pose = self.session_q.get_pose(q_ts, query_sensor_id, self.session_q.proc.alignment_trajectories).to_4x4mat()
            map_pose = self.session_r.get_pose(r_ts, map_sensor_id).to_4x4mat()
            
            T_0to1 = np.linalg.inv(map_pose) @ query_pose
            
            err_t, err_R = compute_pose_error(T_0to1, est_pose)
            all_err_t[query_img].append(err_t)
            all_err_R[query_img].append(err_R)
            detailed_results.append({
                "query": query_img,
                "map": map_img,
                "err_t": float(err_t),
                "err_R": float(err_R)
            })
        map_ = thread_map if self.parallel else lambda f, x: list(map(f, tqdm(x)))
        map_(_worker_fn, range(len(self.pairs)))
        
        all_err_t = self.convert_to_list(all_err_t, self.top)
        all_err_R = self.convert_to_list(all_err_R, self.top)
                
        th_r, th_t = self.Rt_threshold
        moderate_th_r, severe_th_r = self.filter_R_threshold
        recall = np.mean((all_err_R < th_r) & (all_err_t < th_t))

        for item in detailed_results:
            err_R = item["err_R"]
            if moderate_th_r < err_R <= severe_th_r:
                moderate_fails.append(item)
            elif err_R > severe_th_r:
                severe_fails.append(item)

        return all_err_t.mean(), all_err_R.mean(), recall, detailed_results, moderate_fails, severe_fails
    
    
class SingleCrossValidation(CrossValidation):
    method = {
        'name': 'single_image',
        'thresh': 1
    }
    
    
class RigCrossValidation(CrossValidation):
    method = {
        'name': 'rig',
        'thresh': 1,
    }