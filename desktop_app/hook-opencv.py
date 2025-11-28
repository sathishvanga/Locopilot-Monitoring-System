# PyInstaller hook for OpenCV to prevent recursion issues

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs
import os

# Collect OpenCV data files
datas = collect_data_files('cv2')

# Collect OpenCV dynamic libraries
binaries = collect_dynamic_libs('cv2')

# Set environment variable to prevent OpenCV recursion
os.environ['OPENCV_DISABLE_OPENCL'] = '1'

