# coding=utf-8
import sys
import os

# add src path
curr_path = os.path.abspath(__file__)
src_path = os.path.dirname(os.path.dirname(curr_path))
sys.path.append(src_path)
