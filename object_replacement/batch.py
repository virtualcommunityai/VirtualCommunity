import subprocess
import os

names = os.listdir('/project/pi_chuangg_umass_edu/yian/robogen/ljg/Depth-Anything-V2/images/CMU6View')

for name in names:
    if name.startswith('-') or name.startswith('_'):
        continue
    commands = ["python", "./Depth-Anything-V2/code_ljg/main.py", "--panoId", name]
    subprocess.run(commands)