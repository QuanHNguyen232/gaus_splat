# cmsc714-group-prj-Spring25

## Original dataset used in paper:
https://storage.googleapis.com/gresearch/refraw360/360_v2.zip

Quan: I use images from `360_v2/bicycle/images_8` (resolution 1/8 files)

## How to set up environment
Check `./setup_env/README.md`
OR
1. Load modules:
```bash
module load cuda/11.6.2
```
1. Activate Conda env for gau_splatting:
```bash
source /home/nguyqu03/miniconda3/bin/activate gaussian_splatting_1
```

## How to set up batching and running code
Replace files in gaussian_splatting directory with the ones in modified. Request a 20 minute bash session with 2 gpus:

```bash
srun --partition=dpart --qos=medium --gres=gpu:2 --time 0:20:0 --pty bash
# OR
srun --partition=dpart --qos=normal --gres=gpu:a100:2 --time 0:20:0 --pty bash 
# OR
srun --partition=gpu --qos=normal --gres=gpu:a100_1g.5gb:2 --time 0:20:0 --pty bash
```
Note:
```bash
# zaratan (/home/nguyqu03/scratch.cmsc714/) has --qos = normal, scavenger, high, gpu
```
run the following to train with two gpus for resolution 1/8 files, for 200 iterations:

```bash
python multi-run.py -s ./bicycle/ --iterations 200 -r 8 --num_gpu=2
# OR
python train.py -s data/bicycle/ --iterations 200
```

TODO: Fix bugs in call to rasterize c file in line 102 of gaussian_renderer/__init__.py

## How to run profiling:
<!-- Check `./profiling/README.md` -->
Files are on Zaratan `/scratch/zt1/project/cmsc714/shared/3dgs_proj/gaussian-splatting`

### Python
Step 1
```bash
python -m cProfile -o profiling_results/profile_train.prof train.py -s data/bicycle/ --iterations 200
```
Step 2:
```bash
python pyProfile.py
```

### Cuda

Integrate in `train_cudaProfile.py` code. Run CMDs as above to train, just replace `train.py` with `train_cudaProfile.py`.


# Git version control
When you make changes to files in the main repository (but not the submodules), you can simply use:
```bash
git push origin main
```
However, if you make changes to files within the submodule directories, you should use:
```bash
git push --recurse-submodules=on-demand origin main
```
This ensures that both your main repository changes and any submodule changes are pushed to their respective remote repositories.
Here's a practical workflow for making changes:

If you modify files only in the main repository:
```bash
git add .
git commit -m "Update main repository files"
git push origin main
```

If you modify files in both the main repository and submodules:
```bash # First commit changes in submodules
cd path/to/submodule
git add .
git commit -m "Update submodule files"
```
# Return to main repository
```bash
cd ../..
git add .
git commit -m "Update main repository and submodule references"
git push --recurse-submodules=on-demand origin main
```

The --recurse-submodules=on-demand option is valuable because it will:

Check if your submodule changes need to be pushed
Push those changes to the submodule's remote repository
Then push your main repository changes

