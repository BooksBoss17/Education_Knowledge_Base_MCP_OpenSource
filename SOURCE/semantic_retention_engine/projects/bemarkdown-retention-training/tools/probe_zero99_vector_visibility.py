"""Source primitive versus actual crop pixels; diagnostics, no classification rule."""

import hashlib,json,math,sys

from pathlib import Path

import numpy as np

from PIL import Image

import pymupdf

P=Path(__file__).resolve().parents[1];R=P.parents[1]

P=Path(__file__).resolve().parents[1];R=P.parents[1]

def color_support(rgb,color):
    vector=255.-np.asarray(color,dtype=np.float64)*255.
    norm=float(np.dot(vector,vector))
    if norm<1.:return {'white_or_no_contrast':True}
    delta=255.-rgb.astype(np.float64)
    alpha=(delta@vector)/norm
    residual=np.linalg.norm(delta-alpha[...,None]*vector,axis=-1)
    relative=residual/np.maximum(np.linalg.norm(delta,axis=-1),1.)
    nonwhite=(alpha>=.02)&(alpha<=1.15)
    return {'color_rgb01':list(color),'strict_support_pixels':int(np.count_nonzero(nonwhite&(relative<=.15))),
            'broad_support_pixels':int(np.count_nonzero(nonwhite&(relative<=.35))),'total_pixels':int(rgb.shape[0]*rgb.shape[1])}

def near_point_to_box(point,box,eps=1.5):
    return box[0]-eps<=point.x<=box[2]+eps and box[1]-eps<=point.y<=box[3]+eps
