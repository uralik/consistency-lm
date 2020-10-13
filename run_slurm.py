import os
import datetime
import itertools
import getpass

import argparse
from glob import glob
import os
import random

user_folders = {
    'output-dirname': {
        'example_user': '/path_to_checkpoint_folder',
    },
    'base-dir': {
        'example_user': '/path_to_code'
    }
}

current_user = getpass.getuser()

parser = argparse.ArgumentParser()
parser.add_argument('--nhrs', type=int, default=12)
parser.add_argument('--ngpus', type=int, default=1)
parser.add_argument('--ncpus', type=int, default=4)
parser.add_argument('--dryrun', action='store_true')
parser.add_argument('--print-commands', action='store_true')
parser.add_argument('--partition', choices=['partition_name'], default='partition_name')
args = parser.parse_args()

# Global extras
extra_flags = []  # extra flags, specified as {--flagname: None} passed to all jobs
extra_args = []  # extra {--argname: argvalue} args passed to all jobs
extra_name = ''  # extra name suffix appended to `expr`
eval_mode = False # set this to True when want to skip save-base-dir

def get_sweep_model_dirs(sweep_dir_name, extra_folder='wikitextv2'):
    # only takes folders where 'model_best.pt' exists
    dirs = glob(os.path.join(sweep_dir_name, '*', extra_folder))
    valid_dirs = []
    for d in dirs:
        if os.path.exists(os.path.join(d,'model_best.pt')):
            valid_dirs.append(d)
    return valid_dirs
    

name_fields = []

# ======= Experiments


# --- wikitext2 sentencized
DEFAULT_WIKITEXT_SENT_SETTINGS = {
    'script': os.path.join(user_folders['base-dir'][current_user], 'self_terminating/base/train.py'),
    '--dataset-version': 'wikitext_sentencized',
    '--num-samples': '1000',
    '--dataset-path': os.path.join(user_folders['base-dir'][current_user],
                                   'training_data/wikitext2-sentencized.json')
}

DEFAULT_WIKITEXT_SENT_EVAL_SETTINGS = {
    'script': os.path.join(user_folders['base-dir'][current_user], 'self_terminating/base/evaluate.py'),
    '--dataset-version': 'wikitext_sentencized',
    '--dataset-path': os.path.join(user_folders['base-dir'][current_user],
                                   'training_data/wikitext2-sentencized.json')
}






# training: vanilla models

if False:
    GRID = True
    common_settings = DEFAULT_WIKITEXT_SENT_SETTINGS
    common_settings['--num-layers'] = '2'
    common_settings['--optimizer'] = 'adam'
    common_settings['--tie-weights'] = '1'

    num_seeds = 10
    grids = [
        {
            '--hidden-size': ['256'],
            '--dropout': ['0.3'],
            '--lr-anneal': ['0.5'],
            '--rnn-type': ['nn.RNN'],
            '--clip-grad-norm': ['1.0'],
            '--seed': list(map(str, range(num_seeds)))
        },
        {
            '--hidden-size': ['512'],
            '--dropout': ['0.5'],
            '--lr-anneal': ['0.5'],
            '--rnn-type': ['nn.LSTM'],
            '--clip-grad-norm': ['1.0'],
            '--seed': list(map(str, range(num_seeds)))
        },
    ]
    expr = 'wikitext2_sentencized_tied_seeds'



# evaluation: vanilla models (regular decodings) using the whole training sweep

if False:
    GRID = True
    common_settings = DEFAULT_WIKITEXT_SENT_EVAL_SETTINGS
    common_settings['--consistent-sampling'] = '0' 
    eval_mode = True

    sweep_dirs = get_sweep_model_dirs('/path_to_checkpoint_folder/wikitext2_sentencized_tied_seeds/mmdd_hhmm')

    grids = []
    for _dir in sweep_dirs:
        setup = {
                    '--model-load-dir': [_dir],
                    '--dataset-version': ['wikitext_sentencized'],
                    '--output-dir-override': [_dir + '/regular'],
                }
        grids.append(setup)
    expr = 'final_eval_vanilla_regular'


# evaluation: vanilla models (regular decodings) using a single model

if False:
    GRID = True
    common_settings = DEFAULT_WIKITEXT_SENT_EVAL_SETTINGS
    common_settings['--consistent-sampling'] = '0' 
    eval_mode = True

    _dir = '/path_to_checkpoint_folder/wikitext2_sentencized_tied_seeds/mmdd_hhmm/model_run/wikitextv2'
    grids = [
        {
                    '--model-load-dir': [_dir],
                    '--dataset-version': ['wikitext_sentencized'],
                    '--output-dir-override': [_dir + '/regular'],
        },
        ]
    expr = 'final_eval_vanilla_regular'


# evaluation: vanilla models (consistent top-k sampling and consistent nucleus sampling)

if False:
    GRID = True
    common_settings = DEFAULT_WIKITEXT_SENT_EVAL_SETTINGS
    common_settings['--consistent-sampling'] = '1' 
    eval_mode = True

    sweep_dirs = get_sweep_model_dirs('/path_to_checkpoint_folder/wikitext2_sentencized_tied_seeds/mmdd_hhmm')

    grids = []
    for _dir in sweep_dirs:
        setup = {
                    '--model-load-dir': [_dir],
                    '--dataset-version': ['wikitext_sentencized'],
                    '--output-dir-override': [_dir + '/consistent'],
                }
        grids.append(setup)
    expr = 'final_eval_vanilla_consistent'









# training: self-terminating RNN
if False:
    GRID = True
    common_settings = DEFAULT_WIKITEXT_SENT_SETTINGS

    # Best grid search values for LSTM
    common_settings['--rnn-type'] = 'nn.RNN'
    common_settings['--num-layers'] = '2'
    common_settings['--hidden-size'] = '256'
    common_settings['--dropout'] = '0.3'
    common_settings['--embedding-dim'] = '256'
    common_settings['--tie-weights'] = '1'
    common_settings['--clip-grad-norm'] = '1.0'
    common_settings['--lr-anneal'] = '0.5'

    common_settings['--self-terminate'] = '1'

    # Define the experiments
    grids = [
        {
            '--st-epsilon': ['0.00001', '0.0001', '0.001', '0.01']
        }
    ]

    expr = 'wikitext2_sentencized_st_rnn'



# training: self-terminating LSTM
if False:
    GRID = True
    common_settings = DEFAULT_WIKITEXT_SENT_SETTINGS

    # Best grid search values for LSTM
    common_settings['--rnn-type'] = 'nn.LSTM'
    common_settings['--num-layers'] = '2'
    common_settings['--hidden-size'] = '512'
    common_settings['--dropout'] = '0.5'
    common_settings['--embedding-dim'] = '256'
    common_settings['--tie-weights'] = '1'
    common_settings['--clip-grad-norm'] = '1.0'
    common_settings['--lr-anneal'] = '0.5'

    common_settings['--loss-type'] = 'st'
    common_settings['--self-terminate'] = '1'

    # Define the experiments
    grids = [
        {
            '--st-epsilon': ['0.00001', '0.0001', '0.001', '0.01']
        }
    ]

    expr = 'wikitext2_sentencized_st_lstm'






# evaluation: self-terminating RNN and LSTM 
if False:
    GRID = True
    common_settings = DEFAULT_WIKITEXT_SENT_EVAL_SETTINGS
    eval_mode = True

    sweep_dirs = (get_sweep_model_dirs('/path_to_checkpoint_folder/wikitext2_sentencized_st_rnn/mmdd_hhmm') +
                  get_sweep_model_dirs('/path_to_checkpoint_folder/wikitext2_sentencized_st_lstm/mmdd_hhmm'))

    grids = []
    for _dir in sweep_dirs:
        setup = {
                    '--model-load-dir': [_dir],
                    '--dataset-version': ['wikitext_sentencized'],
                }
        grids.append(setup)
    expr = 'final_eval_st'



# ======== BPE tokenized models ========
# In order to train/eval models using BPE tokenized wikitext2, please substitute the data loading argument as:
# '--dataset-path': os.path.join(user_folders['base-dir'][current_user], 'training_data/bpe_wikitext2_raw.json')  
# 
# Make sure to use hyper-parameters specified in BPE grid search table while preparing the model runs (as above).



# ========= Run the job combinations (you shouldn't need to modify/read this code; keeping everything in one file.)
# Setup the base output directory of the form:
#       {args.output_base_dir}/{expr}{extra_name}/{datetime}
# E.g.
#       project_x/output/expr1/0113_0330
now = datetime.datetime.now()
datetime = now.strftime("%m%d_%H%M")
output_dir = os.path.join(user_folders['output-dirname'][current_user], "%s%s" % (expr, extra_name), datetime)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
print("Output Directory: %s" % output_dir)

if GRID:
    # Make combinations of grid values; each combination is a 'job'
    jobs = []
    for grid in grids:
        individual_options = [[{k: v} for v in values]
                              for k, values in grid.items()]
        product_options = list(itertools.product(*individual_options))
        jobs += [{k: v for d in option_set for k, v in d.items()}
                 for option_set in product_options]

    merged_grid = {}
    for grid in grids:
        for key in grid:
            merged_grid[key] = [] if key not in merged_grid else merged_grid[key]
            merged_grid[key] += grid[key]
    name_fields = {key for key in merged_grid if len(set(merged_grid[key])) > 1}

if args.dryrun:
    print("NOT starting {} jobs:".format(len(jobs)))
else:
    print("Starting {} jobs:".format(len(jobs)))

# Do the runs
for job in jobs:
    # Make the name
    name = '%s%s' % (extra_name, job.get('name', expr))
    if len(name_fields) > 0:
        name += '__'
        for k in name_fields:
            name += '__'
            if '--%s' % k in job:
                name += '%s=%s' % (k, str(job['--%s' % k]))
            elif '-%s' % k in job:
                name += '%s=%s' % (k, str(job['-%s' % k]))
            elif k in job:
                if k.startswith('--'):
                    k_ = k[2:]
                elif k.startswith('-'):
                    k_ = k[1:]
                else:
                    k_ = k
                if '/' in job[k] and isinstance(job[k], str):  # folder name there, need to short it, quick hack
                    random_num = random.randint(0,1000)  # we need this to distinguish filenames
                    opt_val = f'masked-folder-name-here-{random_num}'
                    #opt_val = '_slash_'.join(job[k].split('/')[-2:])
                else:
                    opt_val = job[k]
                name += '%s=%s' % (k_, str(opt_val))

    if '/' in name:
        import ipdb; ipdb.set_trace()
        name = name.replace('/', '_slash_')
    print('\n' + name)

    # Pass the name and output directory to the downstream python command.
    if eval_mode is False:
        job['--save-base-dir'] = os.path.join(output_dir, name)
        os.makedirs(job['--save-base-dir'])

    # Make the python command
    script = common_settings.get('script', job.get('script'))
    cmd = ['python', '-u', script]

    for arg, val in common_settings.items():
        if isinstance(val, list):
            cmd.append(arg)
            for item in val:
                cmd.append(item)
        else:
            arg_, val_ = str(arg), str(val)
            if arg_ == 'name' or arg_ == 'script' or arg_ in job:
                continue
            cmd.append(arg_)
            if val is not None:
                cmd.append(val_)

    for arg, val in job.items():
        arg_, val_ = str(arg), str(val)
        if arg_ == 'name' or arg_ == 'script':
            continue
        cmd.append(arg_)
        if val is not None:
            cmd.append(val_)

    for arg, val in extra_args:
        arg_, val_ = str(arg), str(val)
        cmd.append(arg_)
        if val is not None:
            cmd.append(val_)

    for flag in extra_flags:
        flag = str(flag)
        cmd.append(flag)

    cmd = ' '.join(cmd)
    if args.print_commands:
        print(cmd)

    # Make a {name}.slurm file in the {output_dir} which defines this job.
    slurm_script_path = os.path.join(output_dir, '%s.slurm' % name)
    slurm_command = "sbatch %s" % slurm_script_path

    # Make the .slurm file
    with open(slurm_script_path, 'w') as slurmfile:
        slurmfile.write("#!/bin/bash\n")
        slurmfile.write("#SBATCH --job-name" + "=" + name + "\n")
        slurmfile.write("#SBATCH --open-mode=append\n")
        slurmfile.write("#SBATCH --output=%s.out\n" % (os.path.join(output_dir, name)))
        slurmfile.write("#SBATCH --error=%s.err\n" % (os.path.join(output_dir, name)))
        slurmfile.write("#SBATCH --export=ALL\n")
        slurmfile.write("#SBATCH --time=%d:00:00\n" % args.nhrs)
        if args.partition == 'partition_name':
            slurmfile.write("#SBATCH --mem=50G\n")
            slurmfile.write("#SBATCH --gres=gpu:p40:%d\n" % args.ngpus)
        else:
            raise NotImplementedError(args.partition)
        slurmfile.write("#SBATCH -c %d\n" % args.ncpus)
        slurmfile.write('#SBATCH --signal=USR1@60\n')
        slurmfile.write('term_handler () {\n\
    # catch and ignore TERM. we get multiple terms during shutdown, so best\n\
    # to just do nothing\n\
    # but still keep going with the python process\n\
    wait "$CHILD"\n\
}\n\n')
        slurmfile.write('usr1_handler () {\n\
    echo "SLURM signaling preemption/times up (SLURM_PROCID $SLURM_PROCID)." \n\
    kill -s INT "$CHILD"  # send ctrl-c to python\n\
    if {SHOULD_REQUEUE} && [ "$SLURM_PROCID" -eq "0" ]; then\n\
        echo "Waiting 5s and resubmitting..."\n\
        sleep 5\n\
        echo "Resubmitting..."\n\
        scontrol requeue $SLURM_JOB_ID\n\
    fi\n\
    wait "$CHILD"\n\
}\n\n')
        slurmfile.write("trap 'usr1_handler' USR1\n")
        slurmfile.write("trap 'term_handler' TERM\n")
        slurmfile.write("cd " + user_folders['base-dir'][current_user] + '\n')
        slurmfile.write("srun " + cmd)
        slurmfile.write("\n")

    if not args.dryrun:
        os.system("%s &" % slurm_command)

    print("Follow logfile: tail -f %s" % (os.path.join(output_dir, name + '.out')))
