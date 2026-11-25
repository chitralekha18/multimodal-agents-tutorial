# Multimodal Agents Tutorial

A comprehensive tutorial on building and evaluating multimodal machine learning models for predicting learner uncertainty and correctness using text and voice input.

## Overview

This repository contains a complete tutorial for training and evaluating multimodal agents that predict learner state (uncertainty and correctness) across different input modalities. The tutorial covers both the conceptual framework and practical implementation with real benchmark results.

---

## Files in This Repository

### 1. **learner_uncertainty_tutor.ipynb**
   - **What it is:** The main interactive Jupyter notebook tutorial
   - **What you'll learn:** 
     - How to build multimodal models with text and audio input
     - Different objective settings (uncertainty-only, multitask, correctness-only)
     - Model architecture using Qwen with LoRA adapters and audio projectors
     - Training and evaluation procedures
     - How to interpret model outputs
   - **How to use:** Open in Jupyter or JupyterLab and work through each cell sequentially

### 2. **learner_state_multimodal_benchmark.py**
   - **What it is:** The standalone Python benchmark script
   - **What it does:**
     - Trains all 9 benchmark conditions (3 objectives × 3 input modalities)
     - Evaluates models on test data
     - Generates performance metrics (Macro-F1, Balanced Accuracy)
     - Outputs results and model checkpoints
   - **How to use:** 
     - For local testing: `python learner_state_multimodal_benchmark.py`
     - For GPU cluster: Use the SBATCH file (see below)

### 3. **benchmark_summary.md**
   - **What it is:** Results and analysis from a full benchmark run
   - **What it contains:**
     - Details of the 9 benchmark conditions
     - Train/validation/test data split information (70/15/15 by learner)
     - Test results for each model variant
     - Example predictions from each objective
     - Performance comparison between objectives
   - **How to use:** Reference this to understand what full results look like and to validate your own runs

### 4. **run_orcd_batch_state_benchmark.sbatch**
   - **What it is:** An SLURM batch script for running the benchmark on GPU clusters
   - **What it configures:**
     - GPU allocation (1 GPU, 8 CPUs, 32GB memory)
     - Walltime (6 hours)
     - Job logging and output directories
     - Python environment setup
   - **How to use:** See the ORCD section below

---

## Getting Started Locally

### Prerequisites
- Python 3.8+
- PyTorch with GPU support (optional for local testing)
- Jupyter or JupyterLab
- Required Python packages (see Dependencies below)

### Dependencies
Install required packages:
```bash
pip install torch transformers datasets scikit-learn pandas numpy
# For the notebook:
pip install jupyter matplotlib seaborn
```

### Running the Tutorial Locally
1. Start Jupyter:
   ```bash
   jupyter notebook
   ```
2. Open `learner_uncertainty_tutor.ipynb`
3. Work through the cells to understand the concepts

### Running the Benchmark Locally (Small Test)
If you want to run a quick test locally:
```bash
# Set an environment variable to run a smoke test (smaller dataset)
SMOKE_TEST=1 python learner_state_multimodal_benchmark.py
```

---

## Running on ORCD (GPU Cluster)

### Why Use ORCD?
The full benchmark trains 9 models on a GPU, which takes significant time. ORCD (Orange Cluster Resource Deployment) provides:
- GPU access (NVIDIA GPUs for fast training)
- Job scheduling via SLURM
- Shared storage for data and results

### How to Get Access to ORCD

#### If You're at MIT
1. https://orcd-docs.mit.edu/getting-started/ 

2. **Set up SSH access:**
   - Once your account is approved, contact orcd-help@mit.edu to add your SSH public key
   - Typical endpoints: `orcd.mit.edu` or specific cluster addresses

#### If You're Not at MIT
Contact your institution's HPC administrator to request access to a similar GPU cluster (e.g., universities often have equivalent systems). The batch script provided here is generic SLURM, so it should work on other HPC systems with minimal modifications.

### SSH Setup (One-time)
Add this to your `~/.ssh/config`:
```
Host orcd
    HostName orcd.mit.edu
    User your_username
    IdentityFile ~/.ssh/your_key
```

Then connect with: `ssh orcd`

### Preparing Your ORCD Environment

1. **Copy files to ORCD:**
   ```bash
   scp -r /path/to/multimodal-agents-tutorial orcd:~/projects/
   ```

2. **SSH into ORCD:**
   ```bash
   ssh orcd
   cd ~/projects/multimodal-agents-tutorial
   ```

3. **Set up Python environment** (create once):
   ```bash
   # Option 1: Using venv
   python3 -m venv .venv
   source .venv/bin/activate
   pip install torch transformers datasets scikit-learn pandas numpy
   
   # Option 2: Using conda (if available)
   conda create -n multimodal-tutorial python=3.10
   conda activate multimodal-tutorial
   conda install pytorch pytorch-cuda=12.1 -c pytorch
   pip install transformers datasets scikit-learn pandas
   ```

### Running the Benchmark on ORCD

#### Basic Usage
```bash
sbatch run_orcd_batch_state_benchmark.sbatch
```

#### Check Job Status
```bash
squeue -u $USER
```

#### View Output
Once the job completes, check results in `logs/`:
```bash
# Job output log
cat logs/state_benchmark_*.out

# Error log (if any)
cat logs/state_benchmark_*.err
```

#### Advanced: Customize the Batch Job

You can override settings at submission time:

**Run a quick test (smoke test mode):**
```bash
sbatch --export=ALL,SMOKE_TEST=1 run_orcd_batch_state_benchmark.sbatch
```

**Use a different Python binary:**
```bash
sbatch --export=ALL,PYTHON_BIN=/path/to/python run_orcd_batch_state_benchmark.sbatch
```

**Use a different script:**
```bash
sbatch --export=ALL,SCRIPT_PATH=/path/to/script.py run_orcd_batch_state_benchmark.sbatch
```

**Request more resources (e.g., 2 GPUs, 12 hours):**
1. Edit `run_orcd_batch_state_benchmark.sbatch`:
   - Change `#SBATCH --gres=gpu:1` to `#SBATCH --gres=gpu:2`
   - Change `#SBATCH --time=06:00:00` to `#SBATCH --time=12:00:00`
2. Submit as usual: `sbatch run_orcd_batch_state_benchmark.sbatch`

#### Understanding SBATCH Options
- `--job-name`: Name of your job (shows in squeue)
- `--partition`: Queue/partition to submit to (ask your admin for available partitions)
- `--gres=gpu:N`: Number of GPUs to allocate
- `--cpus-per-task`: CPU cores per task
- `--mem`: Total memory per job
- `--time`: Walltime limit (HH:MM:SS)
- `--output/--error`: Log file locations

---

## Workflow

### Local Development
1. Work through the notebook to understand concepts
2. Test changes locally (with `SMOKE_TEST=1` for speed)
3. Debug and iterate

### Full Benchmark Run
1. Set up ORCD access and copy files
2. Configure the SBATCH file if needed
3. Submit: `sbatch run_orcd_batch_state_benchmark.sbatch`
4. Monitor with `squeue -u $USER`
5. Download results to analyze locally

### Analyzing Results
- Check `benchmark_summary.md` for format and interpretation
- Look at logs in `logs/` directory for detailed output
- Model checkpoints are saved during training

---

## Model Architecture Overview

The models use:
- **Backbone:** Qwen (base language model with frozen weights + LoRA adapters)
- **Audio Processing:** Audio projector (when voice is included)
- **Task Heads:** 
  - Uncertainty head (binary classification)
  - Correctness head (binary classification)
  - Depending on objective, one or both heads are used

## Modalities

Models are trained on three input configurations:
- **Text only:** Questions + student responses as text
- **Voice only:** Audio recordings of student responses
- **Voice + Text:** Combined text transcript + audio

## Objectives

Three training objectives are compared:
1. **Uncertainty Only:** Predicts whether student is uncertain (binary)
2. **Multitask (Equal Weight):** Jointly predicts uncertainty + correctness
3. **Correctness Only:** Predicts if answer is correct (binary)

---

## Example Results

From the benchmark summary (full run with 1,745 labeled answers):

### Best Performing Model
- **Objective:** Multitask (uncertainty + correctness)
- **Modality:** Voice + Text
- **Uncertainty F1:** 0.723
- **Correctness F1:** 0.852

---

## Troubleshooting

### "Python binary not executable" error
- Make sure your `.venv` path is correct in the SBATCH file
- Or specify an absolute path with `sbatch --export=ALL,PYTHON_BIN=/path/to/python`

### Job takes too long
- Run `squeue -j [JOB_ID] -o "%.18i %.9P %.8j %.8u %.8T %.10M %.6D %R"` to check job status
- If walltime exceeded, increase `--time` and resubmit
- Try `SMOKE_TEST=1` for a faster test run

### Cannot find data
- Verify all files are copied to ORCD in the same directory structure
- Check paths in the Python script match your ORCD setup

### GPU out of memory
- Reduce batch size in the Python script
- Reduce number of epochs
- Allocate more GPU memory or use `--mem`

---

## Contributing & Citation

If you use this tutorial, please cite the original work and credit the authors involved.

---

## Questions?

For issues with ORCD access or cluster-specific problems, contact your institution's HPC support.

For questions about the models or training procedure, refer to the notebook's detailed explanations.
