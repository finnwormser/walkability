"""
submits a job to slurm for any python file.
"""
import argparse
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def parse_args(args, defaults):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "pyfile",
        type=str,
        # default=defaults['pyfile'],
        help="path of python script"
    )
    parser.add_argument(
        "-p", "--partition",
        type=str,
        default=defaults['partition'],
        help="partition"
    )
    parser.add_argument(
        "--time",
        type=str,
        default=defaults['time'],
        help="running time"
    )
    parser.add_argument(
        "--mem",
        # type=int,
        default=defaults['mem'],
        help="RAM needed"
    )
    parser.add_argument(
        "--test_slurm",
        action="store_true",
        help="create a test slurm sh file"
    )
    parser.add_argument(
        "--shdir",
        type=str,
        default=defaults['shdir'],
        help="dir for .sh files created"
    )

    parser.add_argument(
        "--slurmoutdir",
        type=str,
        default=defaults['slurmoutdir'],
        help="dir for .out files created"
    )

    parser.add_argument(
        "--sleep",
        type=float,
        default=defaults['sleep'],
        help="time in between submitting jobs"
    )
    parser.add_argument(
        "--begin",
        type=str,
        default=defaults['begin'],
        help="job start time"
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=1,
        help="number of gpus"
    )

    args, extra = parser.parse_known_args()

    return args, extra


def makedir_if_needed(dir):
    Path(dir).mkdir(parents=True, exist_ok=True)


def main(args=None):
    if args is None:
        args = sys.argv[1:]

    defaults = {
                    # 'pyfile': 'parser.py', # change to correct location
                    'partition': 'nvgpu',
                    'time': '01-00',
                    'mem': '16G',
                    ## 'test_slurm': False,
                    'shdir': Path.cwd() / 'slurm_shfiles',
                    'slurmoutdir': Path.cwd() / 'slurm_out',
                    'sleep': 2,
                    'begin': 'now',
            }

    args, extra_kwargs_orig = parse_args(args, defaults)
    makedir_if_needed(os.path.join(os.getcwd(), 'slurm_out'))

    pyfile = args.pyfile
    fname = os.path.splitext(os.path.basename(pyfile))[0]
    partition = args.partition
    runtime = args.time
    mem = args.mem
    shdir = Path(args.shdir)
    slurmoutdir = Path(args.slurmoutdir)
    test_slurm = args.test_slurm

    date = datetime.now()
    extra_kwargs = extra_kwargs_orig[:]

    shdir = Path(shdir) / f'{date.strftime("%Y%m%d")}'
    shdir.mkdir(parents=True, exist_ok=True)

    slurmoutdir.mkdir(parents=True, exist_ok=True)

    text = f"""#!/bin/bash
#SBATCH --partition={partition}
#SBATCH --time={runtime}
#SBATCH --mem={mem}
#SBATCH --job-name={fname}
#SBATCH --output={slurmoutdir}/{date.strftime('%Y%m%d')}/slurm_%x_%j_{date.strftime('%Y%m%d-%H%M%S')}.out
#SBATCH --begin={args.begin}"""

    if args.gpu is not None:
        text += """
#SBATCH -G {gpu}""".format(gpu=args.gpu)

    if test_slurm:
        text += """
#SBATCH --test-only"""

    text += """
python {pyfile} {extra_kwargs_str}
""".format(pyfile=pyfile, extra_kwargs_str=' '.join(extra_kwargs))

    sh_file = 'caller_slurm_{fname}_{date}.sh'.format(
        fname=os.path.splitext(fname)[0], date=date.strftime('%Y%m%d-%H%M%S'))

    sh_file = os.path.join(shdir, sh_file)
    with open(sh_file, 'w') as f:
        f.write(text)

    to_run = ['sbatch', sh_file]
    full_run = ' '.join(to_run)

    res = subprocess.run(to_run)
    logging.info('Running:\n {0}'.format(full_run))

    time.sleep(args.sleep)

if __name__ == '__main__':
    main()
