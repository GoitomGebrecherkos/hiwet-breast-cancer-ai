"""Optional one-command BUSI CV training helper for Hiwet."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
print("Run the Streamlit app and use Section 7 -> Install / Train Ultrasound Computer Vision Model.")
print("Command:")
print(f'  {sys.executable} -m streamlit run "{ROOT / "app.py"}"')
