import cv2
import numpy as np
from scipy.interpolate import CubicSpline

import argparse
import sys, os
import time

sys.path.append(os.getcwd())

def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    
    parser.add_argument("--data-dir", default="data/AIS.v1i.yolov8")
    parser.add_argument("--test-sample", action='store_true')
    
    return parser
    
class CentroidsDetector():
    def __init__(self, 
                 n_segments:int = 29):
        '''
            Purpose: Generate Sequence Data
            Args:
                - n_segments: N개의 구간 -> ** Centroids는 N+1개 **
        '''
        
        self.n_segments = n_segments
        
    def __call__(self, binary_mask):
        '''
            Return:
                - Sequence: List[float]
        '''
        
        _, binary_mask = cv2.threshold(binary_mask, 128, 255, cv2.THRESH_BINARY)
        
        # (1) Centroid Initialization
        centroids = self.init_centroids(binary_mask)
        
        top_pt, bottom_pt = self.find_extremes(binary_mask)
        
        # Exception: Cubic Interpolation needs only Increasing sequence
        if top_pt[1] < centroids[0][1]:
            centroids = [top_pt, *centroids]
        if bottom_pt[1] > centroids[-1][1]:
            centroids = [*centroids, bottom_pt] 
        
        # (2) Spline Interpolation -> Slice equal steps -> Sequence
        
        sequence = self.get_sequence(centroids, step=4)
        
        return sequence
        
    def init_centroids(self, binary_mask):
        # Y축 min, max 찾기
        y_coords = np.where(binary_mask.any(axis=1))[0]
        
        ymin, ymax = y_coords.min(), y_coords.max()
        
        factor = (ymax-ymin)/(self.n_segments)
        
        cut_points = [int(ymin + i * factor) for i in range(1, self.n_segments)]
        
        centroids = []
        
        for y_coord in cut_points:
            row = binary_mask[y_coord]
            
            rising_edges = np.where((row[:-1] == 0) & (row[1:] == 255))[0]
            rising_edge = rising_edges[0]
            
            falling_edges = np.where((row[:-1] == 255) & (row[1:] == 0))[0]
            falling_edge = falling_edges[0]
            
            centroids.append([int((rising_edge+falling_edge)//2), y_coord])
            
        return centroids
    
    def find_extremes(self, binary_mask, top_grad: float = 1/2, bottom_grad: float = 1/3):
        
        tl_flag, tr_flag, bl_flag, br_flag = (False for _ in range(0, 4))
        
        for bias in range(-binary_mask.shape[0], binary_mask.shape[0]): # 각 선
            if tr_flag: break
            for x in range(0, binary_mask.shape[0]): # 각 선에 따른 x 좌표
                y = top_grad * x + bias
                if y < 0 or y >= binary_mask.shape[0]:
                    continue
                if 255 in binary_mask[int(y), x]:
                    tr_pt = (x, int(y))
                    # print(f"Top Right: {tr_pt}")
                    tr_flag = True
                    break
        
        for bias in range(-binary_mask.shape[0], binary_mask.shape[0]): # 각 선
            if tl_flag: break
            for x in range(0, binary_mask.shape[0]): # 각 선에 따른 x 좌표
                y = -top_grad * x + bias
                if y < 0 or y >= binary_mask.shape[0]:
                    continue
                if 255 in binary_mask[int(y), x]:
                    tl_pt = (x, int(y))
                    # print(f"Top Left: {tl_pt}")
                    tl_flag = True
                    break
        
        for bias in range(binary_mask.shape[0], -binary_mask.shape[0], -1): # 각 선
            if bl_flag: break
            for x in range(0, binary_mask.shape[0]): # 각 선에 따른 x 좌표
                y = bottom_grad * x + bias
                if y < 0 or y >= binary_mask.shape[0]:
                    continue
                if 255 in binary_mask[int(y), x]:
                    bl_pt = (x, int(y))
                    # print(f"Top Right: {bl_pt}")
                    bl_flag = True
                    break
        
        for bias in range(binary_mask.shape[0], -binary_mask.shape[0], -1): # 각 선
            if br_flag: break
            for x in range(0, binary_mask.shape[0]): # 각 선에 따른 x 좌표
                y = -bottom_grad * x + bias
                if y < 0 or y >= binary_mask.shape[0]:
                    continue
                if 255 in binary_mask[int(y), x]:
                    br_pt = (x, int(y))
                    # print(f"Bottom Right: {br_pt}")
                    br_flag = True
                    break
                
        top_pt = [int((tl_pt[0]+tr_pt[0])/2), int((tl_pt[1]+tr_pt[1])/2)]
        bottom_pt = [int((bl_pt[0]+br_pt[0])/2), int((bl_pt[1]+br_pt[1])/2)]
        
        return top_pt, bottom_pt
    
    def get_sequence(self, centroids, step):
        centroids_np = np.array(centroids)
        
        x = centroids_np[:, 1] # Actually, y in openCV
        y = centroids_np[:, 0] # Actually, x in openCV
        
        cs = CubicSpline(x, y)
        
        top_pt_x, bottom_pt_x = centroids[0][1], centroids[-1][1] # Actually, y in openCV
        seq_x = np.array([x for x in range(top_pt_x, bottom_pt_x, step)]) #
        seq_y = cs(seq_x)
        
        sequence = np.concatenate((seq_y.reshape(-1, 1), seq_x.reshape(-1, 1)), axis=1)
        
        return sequence

def get_mask(path, width:int=640, height:int=640):
    '''
        transform label.txt -> binary mask (3 channels)
    '''
    with open(path) as f:
        pts = f.read().split()
    
    polygon = []
    
    for x, y in zip(pts[1::2], pts[2::2]):
        x, y = float(x) * width, float(y) * height
        polygon.append([int(x), int(y)])
    
    polygon = np.array(polygon)
    
    mask = np.zeros((height, width, 3), dtype=np.uint8)
    mask_value = [255, 255, 255]
    cv2.fillPoly(mask, [polygon], mask_value)
    
    return mask

def test():
    path = rf""
    extractor = CentroidsDetector(n_segments=29)
    mask = get_mask(path)
    sequence = extractor(mask)

def main(args):
    if args.test_sample:
        test()
        sys.exit()
    
    start_time = time.time()
    
    # Sequence Extractor Initialization
    extractor = CentroidsDetector(n_segments=29) # points: n_segments + 1
    
    # Check sequences directory exists
    dirs = ['train', 'test', 'valid']
    
    for dir in dirs:
        os.makedirs(rf"{args.data_dir}/{dir}/sequences", exist_ok=True)
        
    for dir in dirs:
        labels_path = rf"{args.data_dir}/{dir}/labels"
        sequences_path = rf"{args.data_dir}/{dir}/sequences"
        
        for filename in os.listdir(labels_path):
            
            label_path = rf"{labels_path}/{filename}"
            sequence_path = rf"{sequences_path}/{filename}"
            
            binary_mask = get_mask(path=label_path)
            sequence = extractor(binary_mask)
            
            with open(sequence_path, 'w+') as f:
                for idx, coordinate in enumerate(sequence):
                    x, y = coordinate[0], coordinate[1]
                    if idx < len(sequence) - 1:
                        f.write(rf"{x} {y}" + '\n')
                    else:
                        f.write(rf"{x} {y}")
            
            print(rf"{sequence_path} | seq length: {len(sequence)}")
            
    elapsed_time = int(time.time() - start_time)
    print(rf"Time: {elapsed_time//60}m {elapsed_time%60}s")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser('Centroid Detector test', parents=[get_args_parser()])
    
    args = parser.parse_args()
    main(args)