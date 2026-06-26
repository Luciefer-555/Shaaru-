"""
pipeline/shaaru_brain.py
Proxy re-exporting root shaaru_brain.py
"""

import os
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path: sys.path.append(root_dir)

from shaaru_brain import *
from shaaru_brain import answer, _parse_query_intent
