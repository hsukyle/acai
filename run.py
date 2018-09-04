import sys
sys.path.append('~/install/doodad/')
import os
import doodad as dd
import doodad.ec2 as ec2
import doodad.ssh as ssh
import doodad.mount as mount
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('target', type=str, help='filename of bash/python script to execute')
parser.add_argument('--gpu_id', default=1, type=int, help='0: Titan X 1: Titan Xp 2: 1080 Ti (non-display) 3: 1080 Ti display')
parser.add_argument('--debug', action='store_true', help='pdb on error')
parser.add_argument('--image', default='acai', type=str, help='docker image tag')
args = parser.parse_args()

# CHANGE THESE
IMAGE = 'kylehsu/{}:latest'.format(args.image)
OUTPUT_DIR_MOUNT = './log'   # what target script sees
OUTPUT_DIR_LOCAL = './log'
BASE_DIR_LOCAL = './'
BASE_DIR_MOUNT = './'
DOODAD_DIR = '~/install/doodad'
DATA_DIR_MOUNT = './data'
DATA_DIR_LOCAL = './data'
THIS_FILE_DIR = os.path.realpath(os.path.dirname(__file__))

GPU_ID = args.gpu_id


# Local docker
mode_docker = dd.mode.LocalDocker(
    image=IMAGE,
    gpu=True,
    gpu_id=GPU_ID
)

# Run experiment via docker on another machine through SSH
mode_ssh = dd.mode.SSHDocker(
    image=IMAGE,
    credentials=ssh.SSHCredentials(hostname='kelvinxx', username='kylehsu',
                                   identity_file='~/.ssh/id_rsa')
)

# change this based on desired mode!
MY_RUN_MODE = mode_docker

# add mounts as necessary, e.g. code from other directories
mounts = [
    mount.MountLocal(local_dir=DOODAD_DIR, pythonpath=True),  # add doodad
    mount.MountLocal(local_dir=BASE_DIR_LOCAL, mount_point=BASE_DIR_MOUNT),
    mount.MountLocal(local_dir=DATA_DIR_LOCAL, mount_point=DATA_DIR_MOUNT),
    mount.MountLocal(local_dir=OUTPUT_DIR_LOCAL, mount_point=OUTPUT_DIR_MOUNT, output=True),   # output mount
]

print(mounts)

command = 'cd ../mounts && '
if args.target[-2:] == 'sh':
    command += './%s' % args.target
elif args.target[-2:] == 'py':
    if args.debug:
        command += 'python -m pdb -c continue %s' % args.target
    else:
        command += 'python %s' % args.target
else:
    command += args.target
dd.launch_shell(
    command=command,
    mode=MY_RUN_MODE,
    dry=False,
    mount_points=mounts,
    verbose=True
)
