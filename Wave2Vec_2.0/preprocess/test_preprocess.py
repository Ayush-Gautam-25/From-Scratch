import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from preprocess.utils import load_data

train_loader = load_data()

for b in train_loader:
    print(b)