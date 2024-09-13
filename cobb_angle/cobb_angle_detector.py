import cv2
import argparse
import numpy as np
import sys, os
import math
from typing import Optional
import time

sys.path.append(os.getcwd())

def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    
    return parser
    
class CentroidsDetector():
    def __init__(self, 
                 n_segments:int = 20):
        '''
            Args:
                - n_segments: N개의 구간 -> ** Centroids는 N+1개 **
        '''
        
        self.n_segments = n_segments
        
    def __call__(self, binary_mask):
        '''
            Return:
                - centroids: List[float]
        '''
        _, binary_mask = cv2.threshold(binary_mask, 128, 255, cv2.THRESH_BINARY)
        
        centroids = self.init_centroids(binary_mask)
        top_pt, bottom_pt = self.find_extremes(binary_mask)
        
        centroids = [top_pt, *centroids, bottom_pt]
        
        return centroids
        
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

def main(args):
    binary_mask = cv2.imread("./cobb_angle/mask_0002.jpg")
    
    detector = CentroidsDetector(n_segments=19)
    centroids = detector(binary_mask)
    print(centroids)
    print(len(centroids))
    
    for x, y in centroids:
        cv2.line(binary_mask, (x, y), (x, y), color=(0, 0, 255), thickness=3)
    
    cv2.imshow('test', binary_mask)
    
    cv2.waitKeyEx(0)
    cv2.destroyAllWindows()
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser('Centroid Detector test', parents=[get_args_parser()])
    
    args = parser.parse_args()
    main(args)