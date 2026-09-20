"""Import the immutable 0917 toy implementation, including its own v5 modules."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
PARENT=ROOT/'analysis_tools/studies/20260917_nonstationary_q'
sys.path.insert(0,str(PARENT/'v5'))
