#!/usr/bin/env python3
# Auto-generated from learner_uncertainty_tutor.ipynb code cells.
# Run top-to-bottom, similar to notebook execution order.


# ===== Notebook code cell 1 (id: audio-v2-002) =====
# Install the tutorial's dependencies once per local venv and reuse on later runs.
import os, sys, subprocess, importlib.metadata, importlib.util, re
from pathlib import Path

setup_log = Path('/tmp/learner_tutor_setup.log')
venv_root = Path(os.environ.get('VIRTUAL_ENV', Path(sys.executable).resolve().parent.parent))
stamp_dir_candidates = [
    Path.cwd() / '.cache',
    Path.cwd(),
    venv_root,
    Path('/tmp'),
]
deps_stamp = None
for candidate in stamp_dir_candidates:
    try:
        candidate.mkdir(parents=True, exist_ok=True)
        probe = candidate / '.write_test'
        probe.write_text('ok', encoding='utf-8')
        probe.unlink()
        deps_stamp = candidate / '.learner_tutor_deps.v2.ok'
        break
    except Exception:
        continue
if deps_stamp is None:
    raise RuntimeError('Could not find a writable location for dependency stamp file.')

packages = [
    'numpy==2.2.6', 'pandas==2.2.3', 'scipy==1.15.3', 'scikit-learn==1.6.1',
    'transformers==4.51.3', 'peft==0.15.2', 'accelerate==1.6.0',
    'funasr==1.2.6', 'modelscope==1.25.0', 'huggingface_hub>=0.30.2,<1',
    'soundfile', 'ipywidgets', 'matplotlib', 'sentencepiece', 'wandb', 'torchaudio',
]

required_modules = [
    'numpy', 'pandas', 'scipy', 'sklearn', 'transformers', 'peft', 'accelerate',
    'funasr', 'modelscope', 'huggingface_hub', 'soundfile', 'ipywidgets',
    'matplotlib', 'sentencepiece', 'wandb', 'torch', 'torchaudio'
]

def check_missing_modules(modules):
    missing = []
    for module in modules:
        try:
            __import__(module)
        except Exception as exc:
            missing.append(f'{module}: {exc.__class__.__name__}: {exc}')
    return missing

initial_missing = check_missing_modules(required_modules)
need_install = (not deps_stamp.exists()) or bool(initial_missing)
if need_install and importlib.util.find_spec('torch') is None:
    packages.append('torch')

if deps_stamp.exists() and not initial_missing:
    print(f'Using existing environment at {venv_root}; dependency install skipped.')
    installed_stdout = ''
else:
    if deps_stamp.exists() and initial_missing:
        print('Dependency check failed despite existing venv stamp; reinstalling missing packages...')
        for item in initial_missing:
            print(' -', item)

    # Detect CUDA in a separate process so installation does not leave old libraries imported here.
    gpu_probe = (
        "import importlib.util\n"
        "spec = importlib.util.find_spec('torch')\n"
        "if spec is None:\n"
        "    print(0)\n"
        "else:\n"
        "    import torch\n"
        "    print(int(torch.cuda.is_available()))\n"
    )
    gpu_check = subprocess.run(
        [sys.executable, '-c', gpu_probe],
        capture_output=True,
        text=True,
        check=False,
    )
    if gpu_check.returncode != 0:
        print('Warning: CUDA probe failed; continuing without bitsandbytes.')
        if gpu_check.stderr:
            print(gpu_check.stderr)
    elif gpu_check.stdout.strip().splitlines()[-1] == '1':
        packages.append('bitsandbytes>=0.45.5,<0.50')

    installed = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
        '--only-binary=numpy,pandas,scipy,scikit-learn', *packages],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    installed_stdout = installed.stdout
    setup_log.write_text(installed_stdout)
    if installed.returncode != 0:
        print(installed_stdout)
        raise RuntimeError(f'Dependency installation failed. Full log: {setup_log}')
    print('Tutorial dependencies installed successfully.')
    deps_stamp.write_text(
        f'created_by={sys.executable}\npython={sys.version.split()[0]}\n',
        encoding='utf-8'
    )

missing_after = check_missing_modules(required_modules)
if missing_after:
    for item in missing_after:
        print('Missing after install:', item)
    raise RuntimeError(f'Dependency validation failed. Full log: {setup_log}')

if installed_stdout:
    conflicts = [line for line in installed_stdout.splitlines()
                 if ' requires ' in line and 'incompatible' in line]
    for line in conflicts:
        if line.split()[0] in {'gradio', 'gradio-client', 'diffusers'}:
            print(f'Note: preinstalled {line.split()[0]} expects a newer huggingface-hub; this tutorial does not use it.')
        else:
            print('Dependency conflict to review:', line)
    other_errors = [line for line in installed_stdout.splitlines()
                    if 'ERROR:' in line and "pip's dependency resolver" not in line]
    for line in other_errors:
        print(line)

print(f'Full installation log: {setup_log}')
needs_restart = []
for module, package in [('numpy','numpy'), ('pandas','pandas'), ('scipy','scipy'),
                         ('sklearn','scikit-learn'), ('transformers','transformers'), ('peft','peft')]:
    loaded = getattr(sys.modules.get(module), '__version__', None)
    expected = importlib.metadata.version(package)
    if loaded is not None and loaded != expected:
        needs_restart.append(f'{module}: loaded {loaded}, installed {expected}')
if needs_restart:
    print('Restart the session before continuing: ' + '; '.join(needs_restart))
else:
    print('Ready for the next cell, which checks imports and the speech model on the selected device.')

# ===== Notebook code cell 2 (id: audio-v2-003) =====
assert not globals().get('needs_restart', []), 'Restart the session, then run from the top to use the installed package versions.'
from pathlib import Path
import os, re, json, random, hashlib, zipfile, gc, copy, math, importlib.metadata
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, balanced_accuracy_score, confusion_matrix, classification_report
from IPython.display import display, Audio, clear_output
from funasr import AutoModel as SpeechAutoModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

SEED = 42
def seed_all(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
seed_all()
USE_CUDA = torch.cuda.is_available()
DEVICE = 'cuda' if USE_CUDA else 'cpu'
from contextlib import nullcontext

def amp_context():
    return torch.autocast('cuda', dtype=torch.float16) if USE_CUDA else nullcontext()

def clear_device_cache():
    gc.collect()
    if USE_CUDA: torch.cuda.empty_cache()

print('Device:', torch.cuda.get_device_name(0) if USE_CUDA else 'CPU (no CUDA GPU available)')
print('NumPy', np.__version__, '| pandas', pd.__version__)
print("Speech, transformer and adapter imports passed.")

# Check that the speech checkpoint can load in this runtime.
from huggingface_hub import snapshot_download
# Download only the checkpoint and config files needed by FunASR.
# HF's resumable Xet transfer replaces the unusually slow ModelScope transfer.
SPEECH_MODEL_DIR = snapshot_download(
    repo_id='emotion2vec/emotion2vec_base',
    allow_patterns=['emotion2vec_base.pt', 'config.yaml', 'configuration.json'],
)
print('Speech model files downloaded; loading a short feature-extraction check.', flush=True)
speech_probe = SpeechAutoModel(model=SPEECH_MODEL_DIR, device=DEVICE, disable_update=True)
with torch.no_grad():
    probe_result = speech_probe.generate(input=np.zeros(16000,dtype=np.float32), granularity="utterance", extract_embedding=True)
assert np.isfinite(np.asarray(probe_result[0]["feats"])).all()
print("Speech checkpoint feature extraction passed.")
del speech_probe, probe_result
clear_device_cache()

# ===== Notebook code cell 3 (id: audio-v2-005) =====
from pathlib import Path
import json, zipfile, pandas as pd

# Stable Google Drive file ID for pslc_woz_data.zip.
DATASET_FILE_ID = '1CbuV2RfV3QKIZoJCILfL_k2vPUaXZZVs'

# Keep all tutorial assets local to the current working folder.
BASE_DIR = Path.cwd().resolve()
PROJECT = BASE_DIR / 'learner_uncertainty_tutor'
PROJECT.mkdir(parents=True, exist_ok=True)
ARCHIVE_PATH = PROJECT / 'pslc_woz_data.zip'

if not ARCHIVE_PATH.exists() or ARCHIVE_PATH.stat().st_size < 1_000_000_000:
    import importlib.util, subprocess, sys
    if importlib.util.find_spec('gdown') is None:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'gdown'])
    import gdown
    print('Downloading the shared ITSPOKE archive to local project storage...', flush=True)
    gdown.download(id=DATASET_FILE_ID, output=str(ARCHIVE_PATH), quiet=False)
assert ARCHIVE_PATH.exists() and ARCHIVE_PATH.stat().st_size > 1_000_000_000, 'Dataset download is missing or incomplete.'

WORK = PROJECT / 'uncertainty_tutorial'
WORK.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(ARCHIVE_PATH) as z:
    for member in z.infolist():
        dest = (WORK / member.filename).resolve()
        assert dest.is_relative_to(WORK.resolve()), 'Unexpected archive path'
    z.extractall(WORK)
ROOT = WORK / 'PSLC-WOZ-allfiles'
# Save outputs in a consistent local folder.
SAVE = PROJECT / 'tutorial_results'; SAVE.mkdir(parents=True, exist_ok=True)
segments = pd.read_csv(WORK / 'data/transcript_segments.csv')
review_path = SAVE / 'alignment_review.csv'
if not review_path.exists():
    import shutil
    shutil.copyfile(WORK / 'data/alignment_review.csv', review_path)
review = pd.read_csv(review_path, dtype={'turn_id': str, 'problem': str}).fillna('')
review['reviewed'] = review.reviewed.astype(str).str.lower().eq('true')
review['channel'] = review['channel'].astype(object)
print('Human annotation of uncertainty:', json.loads((WORK / 'data/summary.json').read_text())['uncertainty'])
print(f'{review.learner_id.nunique()} learners; {len(review)} labelled turns; {len(segments)} transcript segments')

# ===== Notebook code cell 4 (id: audio-v2-008) =====
from datetime import datetime
import soundfile as sf

# Align using log event windows, never the label values or row numbers.
auto_matches = {}
alignment_counts = []
for learner, sg in segments.groupby('learner_id'):
    windows = []; origin = None
    for path in sorted((ROOT/'borglogs').glob(f'borg_{learner}_*.log')):
        events = [line.split('\t',4) for line in path.read_text(encoding='cp1252').splitlines()]
        events = [e for e in events if len(e)==5]
        stamp = datetime.strptime(next(e[4] for e in events if e[3]=='TIMESTAMP'),'%Y-%m-%d-%H-%M-%S %z')
        if origin is None: origin = stamp
        delta = (stamp-origin).total_seconds()
        for ms, turn, actor, key, value in events:
            if actor=='BORG' and key=='BORG_CHECKED_UNCERTAIN':
                start = [int(e[0])/1000+delta for e in events if e[1]==turn and e[2]=='STUDENT' and e[3]=='TURN_TYPE']
                if start:
                    eid = path.stem.removeprefix('borg_')+'_'+turn
                    windows.append((eid,start[0],int(ms)/1000+delta))
    bounds = np.array([[w[1],w[2]] for w in windows])
    starts, ends = sg.start_seconds.to_numpy(), sg.end_seconds.to_numpy()
    # Require the same unique containment at offsets -1, 0 and +1 seconds.
    matrices = [(starts[:,None]>=bounds[:,0]+offset)&(ends[:,None]<=bounds[:,1]+offset)
                for offset in (-1.,0.,1.)]
    raw = matrices[1]
    coverage = ((raw.sum(1)==1)[:,None]&raw&(raw.sum(0)==1)[None,:]).sum()/len(windows)
    kept = 0
    if coverage >= .8:
        stable = matrices[0]&matrices[1]&matrices[2]
        for j,k in zip(*np.where(stable)):
            if not all(m[j].sum()==1 and m[:,k].sum()==1 for m in matrices): continue
            s = sg.iloc[j]
            if '***' in s.text or s.duration_seconds < .3 or s.duration_seconds > 30: continue
            # Only auto-select a channel when both recorded channels agree closely.
            with sf.SoundFile(ROOT/'soundfiles'/f'{learner}.ogg') as audio:
                assert audio.samplerate==16000
                audio.seek(round(s.start_seconds*16000))
                x=audio.read(round(s.duration_seconds*16000),dtype='float32',always_2d=True)
            if x.shape[1]>1:
                if np.std(x[:,0])<1e-6 or np.std(x[:,1])<1e-6: continue
                if np.corrcoef(x[:,0],x[:,1])[0,1]<.95: continue
            auto_matches[windows[k][0]] = s.segment_id
            kept += 1
    alignment_counts.append({'learner_id':learner,'window_coverage':coverage,'auto_matched':kept})
review['alignment_source'] = np.where(review.reviewed,'human_review','unmatched')
for index,r in review.iterrows():
    if not r.reviewed and r.example_id in auto_matches:
        review.loc[index,['segment_ids','channel','alignment_source']] = [auto_matches[r.example_id],0,'conservative_time_match']
review['accepted'] = review.reviewed | review.alignment_source.eq('conservative_time_match')
review.to_csv(SAVE/'alignment_with_provenance.csv',index=False)
alignment_report = pd.DataFrame(alignment_counts)
alignment_report.to_csv(SAVE/'alignment_diagnostics.csv',index=False)
print(f'{review.accepted.sum()} accepted examples; {review.loc[review.accepted,"learner_id"].nunique()} learners')
print('Automatic matches are provisional, not human-verified. Review them before reporting research results.')
from IPython.display import Markdown

display(Markdown("""
**How to read the table**

- **learner_id:** `pslc01-c1` means participant 01 in this corpus, assigned to experimental condition 1. The suffix is a study group, not a learner-state label: c1 used uncertainty-based extra tutoring, c2 used correctness only, and c3 added extra tutoring to some correct answers as a control. These treatments applied to problem 58; problem 59 used the normal tutor.
- **window_coverage:** the fraction of this learner's labelled response windows that contain exactly one uniquely assigned transcript segment, before the stricter filters. A response window runs from the logged student-turn event to the human annotation event. For example, 0.90 means 90% of windows have a unique timing match at the assumed clock offset. It is a timing check, not 90% verified accuracy. We skip automatic matching for learners below 0.80.
- **auto_matched:** the number of examples that survive all automatic checks: a unique match at each of the three timing offsets, acceptable duration, no marked overlap, and sufficiently similar stereo channels. These are proposed audio–transcript–label matches, not human-verified ones. A high window coverage can still leave fewer examples after these checks.
"""))
display(alignment_report.style.format({'window_coverage': '{:.1%}'}))

seg_lookup=segments.set_index('segment_id')
def read_interval(learner,start,end):
    with sf.SoundFile(ROOT/'soundfiles'/f'{learner}.ogg') as f:
        sr=f.samplerate; f.seek(max(0,round(start*sr)))
        x=f.read(round((end-max(0,start))*sr),dtype='float32',always_2d=True)
    return x,sr

def inspect_match(example_id):
    """Show the exact matched answer, plus surrounding audio for checking context."""
    r = review.set_index('example_id').loc[example_id]
    assert r.accepted, "Choose an accepted match from review."
    ss = seg_lookup.loc[str(r.segment_ids).split(';')].sort_values('start_seconds')
    start, end = float(ss.start_seconds.min()), float(ss.end_seconds.max())
    print(f"Example: {example_id} | Learner: {r.learner_id} | Problem: {r.problem} | Turn: {r.turn_id}")
    print('Question:', r.prompt)
    print('Transcript:', ' '.join(ss.text))
    print('Human uncertainty label:', 'true — expressed uncertainty' if int(r.uncertain) else 'false — no expressed uncertainty')
    print('Human correctness label:', 'correct' if int(r.correct) else 'incorrect')
    print(f"Match source: {r.alignment_source} | Recording interval: {start:.2f}–{end:.2f} seconds")
    print('Matched answer — check that the spoken words match the transcript:')
    x, sr = read_interval(r.learner_id, start, end)
    display(Audio(x[:, int(r.channel)], rate=sr))
    print('Surrounding audio — includes up to 15 seconds before the answer; the question may begin earlier:')
    x, sr = read_interval(r.learner_id, max(0, start-15), end+1)
    display(Audio(x[:, int(r.channel)], rate=sr))

# ===== Notebook code cell 5 (id: audio-v2-009) =====
# Listen to one example of each uncertainty label, when both are available.
from IPython.display import Markdown

display(Markdown("""
**Listen and check two matches**

For each example, play the answer and compare it with the transcript. Then use the surrounding audio and printed question to check that this is the right turn. The labels shown are the human annotations from the matched log entry, not model predictions. Consider whether the uncertainty judgment seems plausible; listening alone cannot prove that the label was attached to the right turn.

These examples remain provisional. Playing them does not mark them as reviewed. To inspect another match, call `inspect_match(example_id)` with an ID from `review`.
"""))
preview_pool = review.loc[review.accepted].sort_values('example_id').copy()
preview_pool['preview_uncertain'] = preview_pool.uncertain.astype(int)
preview_ids = preview_pool.groupby('preview_uncertain', sort=True).head(1).example_id.tolist()[:2]
if len(preview_ids) < 2:
    preview_ids += preview_pool.loc[~preview_pool.example_id.isin(preview_ids), 'example_id'].head(2-len(preview_ids)).tolist()
assert preview_ids, 'No accepted matches to inspect. Check the alignment output above.'
for number, example_id in enumerate(preview_ids, 1):
    print(f"\nEXAMPLE {number}")
    inspect_match(example_id)

# ===== Notebook code cell 6 (id: audio-v2-011) =====
# Smoke mode uses at most two answers per learner. Set False for all accepted examples.
# Override with env var: SMOKE_TEST=0 (full) or SMOKE_TEST=1 (smoke).
SMOKE_TEST = os.getenv('SMOKE_TEST', '1').strip().lower() not in {'0', 'false', 'no'}
RUN_MODE = 'smoke' if SMOKE_TEST else 'full'

def clean_text(text):
    text = re.sub(r'<[^>]*>', ' ', str(text))
    text = re.sub(r"[^\w\s']", ' ', text.lower())
    return re.sub(r'\s+', ' ', text).strip()

rows = []
for r in review[review.accepted].itertuples():
    ss = seg_lookup.loc[r.segment_ids.split(';')].sort_values('start_seconds')
    assert ss.learner_id.eq(r.learner_id).all()
    text = clean_text(' '.join(ss.text))
    if not text: continue
    # Preserve pauses inside a turn; do not stitch away the gaps.
    start, end = float(ss.start_seconds.min()), float(ss.end_seconds.max())
    assert end > start and end-start <= 120, 'Review an unusually long merged turn'
    rows.append(dict(example_id=r.example_id, learner_id=r.learner_id, condition=r.learner_id[-2:],
                     transcript=text, prompt=r.prompt, start=start, end=end, channel=int(r.channel),
                     uncertain=int(r.uncertain), correct=int(r.correct)))
df = pd.DataFrame(rows)
assert len(df), 'No accepted pairs. Inspect the alignment diagnostics in section 3.'
learners = df[['learner_id','condition']].drop_duplicates()
assert len(learners) >= 30, 'Review at least 30 learners across conditions for this three-way comparison.'
train_ids, held_ids = train_test_split(learners, test_size=.30, random_state=SEED, stratify=learners.condition)
val_ids, test_ids = train_test_split(held_ids, test_size=.50, random_state=SEED, stratify=held_ids.condition)
split_map = {x: name for name, table in [('train',train_ids),('val',val_ids),('test',test_ids)] for x in table.learner_id}
df['split'] = df.learner_id.map(split_map)
assert df.groupby('learner_id').split.nunique().max() == 1
assert df.groupby('split').uncertain.nunique().eq(2).all(), 'Each split needs both classes.'

accepted_count = len(df)
if SMOKE_TEST:
    # Preserve the learner split; keep at most two answers per learner.
    df = pd.concat([g.sample(n=min(2, len(g)), random_state=SEED)
                    for _, g in df.groupby('learner_id')], ignore_index=True)
    assert df.groupby('split').uncertain.nunique().eq(2).all(), 'Each split needs both uncertainty labels.'

df = df.reset_index(drop=True)
df.to_csv(SAVE/'examples.csv', index=False)

from IPython.display import Markdown
mode = 'SMOKE TEST — small subset for checking code' if SMOKE_TEST else 'FULL DATA — all accepted examples'
print(f"{mode}: {len(df):,} answers from {df.learner_id.nunique()} learners.")
if SMOKE_TEST:
    print(f"Using at most two answers per learner out of {accepted_count:,} accepted examples. These scores are not performance estimates.")
display(Markdown("""
Each **example is one learner answer**: its audio clip, transcript, tutor question and human uncertainty label. We split by learner so that nobody appears in more than one group.

**Training** fits the model; **validation** helps choose the training settings; **test** evaluates the selected model on unseen learners. The table shows only the examples selected for this run. “Uncertain answers (%)” is the percentage labelled uncertain by the human annotator, not a model score.
"""))
summary = df.groupby('split').agg(
    learners=('learner_id', 'nunique'), answers=('example_id', 'size'),
    uncertain_answers=('uncertain', 'sum')) .reindex(['train','val','test'])
summary['nonuncertain_answers'] = summary.answers - summary.uncertain_answers
summary['uncertain_percent'] = 100 * summary.uncertain_answers / summary.answers
summary.index = ['Training', 'Validation', 'Test']
summary.index.name = 'Data group'
summary = summary.rename(columns={
    'learners':'Learners', 'answers':'Answers (examples)',
    'uncertain_answers':'Uncertain answers', 'nonuncertain_answers':'Non-uncertain answers',
    'uncertain_percent':'Uncertain answers (%)'})
display(summary.style.format({'Uncertain answers (%)':'{:.1f}%'}))
print('The roughly 70/15/15 split applies to learners. Answer counts differ because learners spoke different numbers of times.')
print('After changing SMOKE_TEST, rerun Steps 5 onward; earlier model outputs do not describe the newly selected data.')

# ===== Notebook code cell 7 (id: audio-v2-013) =====
from tqdm.auto import tqdm
import torch.nn.functional as F
K, AUDIO_DIM = 8, 768
CACHE = SAVE / 'audio_tokens_v2'; CACHE.mkdir(exist_ok=True)
# Keep new model outputs separate from the earlier pooled-fusion experiment.
RUN_SAVE = SAVE / ('audio_tokens_v2_' + RUN_MODE); RUN_SAVE.mkdir(exist_ok=True)
SPEECH_ID = SPEECH_MODEL_DIR
speech = SpeechAutoModel(model=SPEECH_ID, device=DEVICE, disable_update=True)
speech.model.eval()
for parameter in speech.model.parameters(): parameter.requires_grad = False

def audio_tokens(r):
    signature = json.dumps([r.learner_id,float(r.start),float(r.end),int(r.channel),SPEECH_ID,'frame-bins-v2',K])
    path = CACHE / (hashlib.sha256(signature.encode()).hexdigest()[:24]+'.npy')
    if path.exists():
        cached = np.load(path)
        assert cached.shape == (K,AUDIO_DIM) and np.isfinite(cached).all()
        return cached
    x,sr = read_interval(r.learner_id,r.start,r.end)
    assert sr == 16000
    waveform = x[:,int(r.channel)]
    frames = []
    # Array splitting avoids silently dropping a short remainder.
    chunks = np.array_split(waveform, max(1,math.ceil(len(waveform)/(20*sr))))
    for chunk in chunks:
        with torch.no_grad():
            result = speech.generate(input=chunk,granularity='frame',extract_embedding=True)
        feature = result[0]['feats']
        if torch.is_tensor(feature): feature = feature.detach().cpu().numpy()
        feature = np.asarray(feature,dtype=np.float32)
        assert feature.ndim == 2 and feature.shape[1] == AUDIO_DIM and np.isfinite(feature).all()
        frames.append(feature)
    frames = np.concatenate(frames,axis=0)
    assert len(frames) >= K, 'Not enough frames for eight nonempty bins.'
    bins = np.stack([part.mean(0) for part in np.array_split(frames,K)]).astype('float32')
    np.save(path,bins)
    return bins

A = np.stack([audio_tokens(r) for r in tqdm(list(df.itertuples()),desc='Encoding learner answers')])
assert A.shape == (len(df),K,AUDIO_DIM)
print(f'{len(df):,} answers × {K} ordered bins × {AUDIO_DIM} audio features')
del speech; clear_device_cache()

# ===== Notebook code cell 8 (id: audio-v2-015) =====
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
MODEL_ID = 'Qwen/Qwen2.5-1.5B-Instruct'
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
# Use full causal attention; no sliding-window attention is needed here.
config = AutoConfig.from_pretrained(MODEL_ID)
config.use_sliding_window = False
config.sliding_window = None
load_options = dict(config=config, attn_implementation='sdpa', low_cpu_mem_usage=True)
quantization_mode = 'none'
if USE_CUDA:
    load_options.update(device_map={'': 0}, torch_dtype=torch.float16,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16))
else:
    load_options.update(device_map={'': 'cpu'}, torch_dtype=torch.float32)

try:
    base = AutoModelForCausalLM.from_pretrained(MODEL_ID, **load_options)
    if USE_CUDA:
        quantization_mode = '4-bit'
except RuntimeError as exc:
    message = str(exc)
    cuda_bnb_mismatch = ('CUDA VERSION MISMATCH' in message) or ('Configured CUDA binary not found' in message)
    if USE_CUDA and cuda_bnb_mismatch:
        print('bitsandbytes CUDA mismatch detected; retrying without 4-bit quantization.')
        fallback_options = dict(config=config, attn_implementation='sdpa', low_cpu_mem_usage=True)
        fallback_options.update(device_map={'': 0}, torch_dtype=torch.float16)
        base = AutoModelForCausalLM.from_pretrained(MODEL_ID, **fallback_options)
        quantization_mode = 'fp16'
    else:
        raise

if USE_CUDA and quantization_mode == '4-bit':
    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False})
else:
    base.requires_grad_(False)
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    base.enable_input_require_grads()
if USE_CUDA:
    print('Qwen loading mode:', f'{quantization_mode} on GPU')
else:
    print('Qwen loading mode: float32 on CPU')
base.config.use_cache = False
llm = get_peft_model(base,LoraConfig(r=8,lora_alpha=16,lora_dropout=.05,
    target_modules=['q_proj','v_proj'],bias='none',task_type='CAUSAL_LM'))
assert llm.config.hidden_size == 1536
lora_count, qwen_count = llm.get_nb_trainable_parameters()
print(f'LoRA inside Qwen: {lora_count:,} trainable parameters ({100*lora_count/qwen_count:.4f}% of {qwen_count:,}).')
print('Qwen’s original weights stay frozen. The projector and prediction head are counted in Step 8.')
tr = np.flatnonzero(df.split.eq('train')); va = np.flatnonzero(df.split.eq('val')); te = np.flatnonzero(df.split.eq('test'))
y = df.uncertain.to_numpy(dtype=np.float32)
# Each audio feature uses training data only for normalization, across answers and bins.
am = A[tr].mean(axis=(0,1)); ast = A[tr].std(axis=(0,1)).clip(.01)
a = torch.tensor((A-am)/ast,dtype=torch.float32)
labels = torch.tensor(y)
np.savez(RUN_SAVE/'normalization.npz',am=am,ast=ast)
df.to_csv(RUN_SAVE/'examples.csv',index=False)


# ===== Benchmark extension: modalities x model objectives =====
# Modalities: text_only, voice_only, voice_text
# Models:
#   A = uncertainty-only
#   B = multi-task (uncertainty + correctness, equal-weight loss)
#   C = correctness-only

MAX_Q, MAX_A = 160, 128

tokenizer.truncation_side = "right"
prefix = tokenizer('Estimate learner state. Question: ', add_special_tokens=False)['input_ids']
middle = tokenizer('\nLearner answer: ', add_special_tokens=False)['input_ids']
suffix = tokenizer('\nState estimate:', add_special_tokens=False)['input_ids']

def sequence(row):
    question_ids = tokenizer(
        row.prompt or 'Question unavailable',
        add_special_tokens=False,
        truncation=True,
        max_length=MAX_Q,
    )['input_ids']
    answer_ids = tokenizer(
        row.transcript,
        add_special_tokens=False,
        truncation=True,
        max_length=MAX_A,
    )['input_ids']
    return prefix + question_ids + middle + answer_ids + suffix

seqs = [sequence(r) for r in df.itertuples()]
voice_stub_ids = tokenizer('Voice only input.', add_special_tokens=False)['input_ids'] or [tokenizer.eos_token_id]

labels_uncertainty = torch.tensor(df.uncertain.to_numpy(dtype=np.float32), device=DEVICE)
labels_correctness = torch.tensor(df.correct.to_numpy(dtype=np.float32), device=DEVICE)


def build_batch(indices, modality):
    indices = [int(i) for i in indices]

    if modality == 'voice_only':
        ids = torch.tensor([voice_stub_ids for _ in indices], dtype=torch.long, device=DEVICE)
        mask = torch.ones_like(ids)
    else:
        max_len = max(len(seqs[i]) for i in indices)
        ids = torch.full((len(indices), max_len), tokenizer.pad_token_id, dtype=torch.long, device=DEVICE)
        mask = torch.zeros_like(ids)
        for j, i in enumerate(indices):
            token_ids = torch.tensor(seqs[i], dtype=torch.long, device=DEVICE)
            ids[j, :len(token_ids)] = token_ids
            mask[j, :len(token_ids)] = 1

    audio_tensor = a[indices].to(DEVICE) if modality in {'voice_only', 'voice_text'} else None
    batch = {
        'ids': ids,
        'mask': mask,
        'a': audio_tensor,
        'y_uncertainty': labels_uncertainty[indices],
        'y_correctness': labels_correctness[indices],
    }
    return batch


class StateModel(nn.Module):
    def __init__(self, llm, use_text, use_audio, objective):
        super().__init__()
        self.llm = llm
        self.use_text = bool(use_text)
        self.use_audio = bool(use_audio)
        self.objective = objective

        hidden = llm.config.hidden_size
        self.audio_projector = None
        if self.use_audio:
            self.register_buffer('audio_scale', self.llm.get_input_embeddings().weight.detach().float().std())
            self.audio_projector = nn.Sequential(
                nn.Linear(AUDIO_DIM, 256),
                nn.GELU(),
                nn.Linear(256, hidden),
                nn.LayerNorm(hidden),
            ).to(DEVICE)

        self.head_uncertainty = nn.Linear(hidden, 1).to(DEVICE) if objective in {'A', 'B'} else None
        self.head_correctness = nn.Linear(hidden, 1).to(DEVICE) if objective in {'B', 'C'} else None

    def _compose_inputs(self, ids, mask, audio):
        if self.use_text:
            text_embeds = self.llm.get_input_embeddings()(ids)
            text_mask = mask
        else:
            # Keep one neutral token so the decoder has a valid anchor state.
            stub_ids = ids[:, :1]
            text_embeds = self.llm.get_input_embeddings()(stub_ids)
            text_mask = mask[:, :1]

        if self.use_audio:
            assert audio is not None and audio.shape[1:] == (K, AUDIO_DIM)
            projected = self.audio_projector(audio).to(text_embeds.dtype)
            projected = projected * self.audio_scale
            inputs_embeds = torch.cat([projected, text_embeds], dim=1)
            attention_mask = torch.cat([mask.new_ones((len(ids), K)), text_mask], dim=1)
        else:
            inputs_embeds, attention_mask = text_embeds, text_mask

        return inputs_embeds, attention_mask

    def forward(self, ids, mask, a=None):
        inputs_embeds, attention_mask = self._compose_inputs(ids, mask, a)
        hidden_states = self.llm.get_base_model().model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        ).last_hidden_state
        last_index = attention_mask.sum(1) - 1
        pooled = hidden_states[torch.arange(len(ids), device=DEVICE), last_index].float()

        outputs = {}
        if self.head_uncertainty is not None:
            outputs['uncertainty'] = self.head_uncertainty(pooled).squeeze(-1)
        if self.head_correctness is not None:
            outputs['correctness'] = self.head_correctness(pooled).squeeze(-1)
        return outputs


def reset_lora_state(saved_state):
    with torch.no_grad():
        for name, param in llm.named_parameters():
            if name in saved_state:
                param.copy_(saved_state[name].to(device=param.device, dtype=param.dtype))


def snapshot_trainable(model):
    return {name: p.detach().cpu().clone() for name, p in model.named_parameters() if p.requires_grad}


def restore_snapshot(model, state):
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in state:
                param.copy_(state[name].to(device=param.device, dtype=param.dtype))


def binary_metrics(y_true, probabilities, threshold=0.5):
    predictions = (np.asarray(probabilities) >= threshold).astype(int)
    return {
        'macro_f1': f1_score(y_true, predictions, average='macro', zero_division=0),
        'balanced_accuracy': balanced_accuracy_score(y_true, predictions),
    }


def evaluate_model(model, indices, modality):
    model.eval()
    outputs = {'uncertainty': [], 'correctness': []}
    with torch.no_grad():
        for idx in indices:
            batch = build_batch([int(idx)], modality)
            with amp_context():
                logits = model(ids=batch['ids'], mask=batch['mask'], a=batch['a'])
            if 'uncertainty' in logits:
                outputs['uncertainty'].append(torch.sigmoid(logits['uncertainty']).float().cpu().item())
            if 'correctness' in logits:
                outputs['correctness'].append(torch.sigmoid(logits['correctness']).float().cpu().item())

    if not outputs['uncertainty']:
        outputs['uncertainty'] = None
    if not outputs['correctness']:
        outputs['correctness'] = None
    return outputs


def positive_weight(target_values):
    positives = float(target_values.sum())
    negatives = float(len(target_values) - positives)
    if positives <= 0:
        return torch.tensor(1.0, device=DEVICE)
    return torch.tensor(max(1.0, negatives / positives), device=DEVICE)


RUN_MODE = 'smoke' if globals().get('SMOKE_TEST', True) else 'full'
RUN_SAVE = SAVE / ('state_benchmark_' + RUN_MODE)
RUN_SAVE.mkdir(parents=True, exist_ok=True)

print(f'Running state benchmark ({RUN_MODE}) on {"GPU" if USE_CUDA else "CPU"}.')

EPOCHS, ACCUM = (1 if RUN_MODE == 'smoke' else 3), 8
model_specs = {
    'A': {'name': 'uncertainty_only', 'tasks': ('uncertainty',)},
    'B': {'name': 'multitask_equal_weight', 'tasks': ('uncertainty', 'correctness')},
    'C': {'name': 'correctness_only', 'tasks': ('correctness',)},
}
modalities = {
    'text_only': {'use_text': True, 'use_audio': False},
    'voice_only': {'use_text': False, 'use_audio': True},
    'voice_text': {'use_text': True, 'use_audio': True},
}

initial_lora_state = {name: p.detach().cpu().clone() for name, p in llm.named_parameters() if p.requires_grad}

uncertainty_pos_weight = positive_weight(labels_uncertainty[tr].detach().cpu().numpy())
correctness_pos_weight = positive_weight(labels_correctness[tr].detach().cpu().numpy())

checkpoints = {}
validation_summary = []

for model_key, model_info in model_specs.items():
    for modality_name, modality_cfg in modalities.items():
        run_id = f'{model_info["name"]}__{modality_name}'
        print(f'\n=== Training {run_id} ===', flush=True)
        seed_all()
        reset_lora_state(initial_lora_state)

        candidate = StateModel(
            llm=llm,
            use_text=modality_cfg['use_text'],
            use_audio=modality_cfg['use_audio'],
            objective=model_key,
        )

        params = [p for p in candidate.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=0.01)
        scaler = torch.amp.GradScaler('cuda', enabled=USE_CUDA)

        best_state, best_selection_score = None, -1.0
        best_val_uncertainty, best_val_correctness = np.nan, np.nan

        for epoch in range(EPOCHS):
            candidate.train()
            llm.train()
            order = np.random.default_rng(SEED + epoch).permutation(tr)
            optimizer.zero_grad(set_to_none=True)
            running_loss = 0.0

            for step, idx in enumerate(order):
                batch = build_batch([int(idx)], modality_name)
                group_size = min(ACCUM, len(order) - (step // ACCUM) * ACCUM)
                with amp_context():
                    logits = candidate(ids=batch['ids'], mask=batch['mask'], a=batch['a'])

                    if model_key == 'A':
                        loss_uncertainty = nn.functional.binary_cross_entropy_with_logits(
                            logits['uncertainty'].float(),
                            batch['y_uncertainty'],
                            pos_weight=uncertainty_pos_weight,
                        )
                        loss = loss_uncertainty
                    elif model_key == 'B':
                        loss_uncertainty = nn.functional.binary_cross_entropy_with_logits(
                            logits['uncertainty'].float(),
                            batch['y_uncertainty'],
                            pos_weight=uncertainty_pos_weight,
                        )
                        loss_correctness = nn.functional.binary_cross_entropy_with_logits(
                            logits['correctness'].float(),
                            batch['y_correctness'],
                            pos_weight=correctness_pos_weight,
                        )
                        loss = 0.5 * (loss_uncertainty + loss_correctness)
                    else:
                        loss_correctness = nn.functional.binary_cross_entropy_with_logits(
                            logits['correctness'].float(),
                            batch['y_correctness'],
                            pos_weight=correctness_pos_weight,
                        )
                        loss = loss_correctness

                scaler.scale(loss / group_size).backward()
                running_loss += float(loss.item())

                if (step + 1) % ACCUM == 0 or (step + 1) == len(order):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(params, 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

            val_outputs = evaluate_model(candidate, va, modality_name)

            val_uncertainty_metric = np.nan
            if val_outputs['uncertainty'] is not None:
                val_uncertainty_metric = binary_metrics(
                    y[va].astype(int),
                    val_outputs['uncertainty'],
                )['macro_f1']

            val_correctness_metric = np.nan
            if val_outputs['correctness'] is not None:
                val_correctness_metric = binary_metrics(
                    df.iloc[va].correct.to_numpy(dtype=int),
                    val_outputs['correctness'],
                )['macro_f1']

            if model_key == 'A':
                selection_score = val_uncertainty_metric
            elif model_key == 'B':
                selection_score = np.nanmean([val_uncertainty_metric, val_correctness_metric])
            else:
                selection_score = val_correctness_metric

            print(
                f'{run_id} epoch {epoch + 1}/{EPOCHS} '
                f'loss={running_loss / len(order):.4f} '
                f'val_uncertainty_macro_f1={val_uncertainty_metric:.4f} '
                f'val_correctness_macro_f1={val_correctness_metric:.4f}',
                flush=True,
            )

            if selection_score > best_selection_score:
                best_selection_score = selection_score
                best_state = snapshot_trainable(candidate)
                best_val_uncertainty = val_uncertainty_metric
                best_val_correctness = val_correctness_metric

        restore_snapshot(candidate, best_state)
        checkpoint_path = RUN_SAVE / f'{run_id}.pt'
        torch.save(best_state, checkpoint_path)
        checkpoints[run_id] = str(checkpoint_path)

        validation_summary.append({
            'run_id': run_id,
            'model': model_info['name'],
            'model_key': model_key,
            'modality': modality_name,
            'selection_score': float(best_selection_score),
            'val_macro_f1_uncertainty': float(best_val_uncertainty) if not np.isnan(best_val_uncertainty) else np.nan,
            'val_macro_f1_correctness': float(best_val_correctness) if not np.isnan(best_val_correctness) else np.nan,
        })

        del candidate, params, optimizer
        clear_device_cache()

# Test-time evaluation
rows = []
test_predictions = df.iloc[te][['example_id', 'learner_id', 'uncertain', 'correct']].copy()

for model_key, model_info in model_specs.items():
    for modality_name, modality_cfg in modalities.items():
        run_id = f'{model_info["name"]}__{modality_name}'
        candidate = StateModel(
            llm=llm,
            use_text=modality_cfg['use_text'],
            use_audio=modality_cfg['use_audio'],
            objective=model_key,
        )
        restore_snapshot(candidate, torch.load(checkpoints[run_id], map_location='cpu', weights_only=True))

        outputs = evaluate_model(candidate, te, modality_name)

        if outputs['uncertainty'] is not None:
            test_predictions[f'{run_id}__uncertainty'] = outputs['uncertainty']
            uncertainty_metrics = binary_metrics(y[te].astype(int), outputs['uncertainty'])
            rows.append({
                'run_id': run_id,
                'model': model_info['name'],
                'model_key': model_key,
                'modality': modality_name,
                'target': 'uncertainty',
                'test_macro_f1': float(uncertainty_metrics['macro_f1']),
                'test_balanced_accuracy': float(uncertainty_metrics['balanced_accuracy']),
            })

        if outputs['correctness'] is not None:
            test_predictions[f'{run_id}__correctness'] = outputs['correctness']
            correctness_metrics = binary_metrics(df.iloc[te].correct.to_numpy(dtype=int), outputs['correctness'])
            rows.append({
                'run_id': run_id,
                'model': model_info['name'],
                'model_key': model_key,
                'modality': modality_name,
                'target': 'correctness',
                'test_macro_f1': float(correctness_metrics['macro_f1']),
                'test_balanced_accuracy': float(correctness_metrics['balanced_accuracy']),
            })

        del candidate
        clear_device_cache()

# Baselines reported for both targets.
majority_uncertainty = int(y[tr].mean() >= 0.5)
majority_correctness = int(df.iloc[tr].correct.mean() >= 0.5)

baseline_uncertainty = np.full(len(te), majority_uncertainty, dtype=float)
baseline_correctness = np.full(len(te), majority_correctness, dtype=float)

rows.append({
    'run_id': 'majority_baseline',
    'model': 'majority_baseline',
    'model_key': 'baseline',
    'modality': 'na',
    'target': 'uncertainty',
    'test_macro_f1': float(binary_metrics(y[te].astype(int), baseline_uncertainty)['macro_f1']),
    'test_balanced_accuracy': float(binary_metrics(y[te].astype(int), baseline_uncertainty)['balanced_accuracy']),
})
rows.append({
    'run_id': 'majority_baseline',
    'model': 'majority_baseline',
    'model_key': 'baseline',
    'modality': 'na',
    'target': 'correctness',
    'test_macro_f1': float(binary_metrics(df.iloc[te].correct.to_numpy(dtype=int), baseline_correctness)['macro_f1']),
    'test_balanced_accuracy': float(binary_metrics(df.iloc[te].correct.to_numpy(dtype=int), baseline_correctness)['balanced_accuracy']),
})

validation_df = pd.DataFrame(validation_summary).sort_values(['model_key', 'modality'])
results_df = pd.DataFrame(rows).sort_values(['target', 'model_key', 'modality'])

validation_path = RUN_SAVE / 'validation_summary.csv'
results_path = RUN_SAVE / 'test_metrics.csv'
predictions_path = RUN_SAVE / 'test_predictions.csv'

validation_df.to_csv(validation_path, index=False)
results_df.to_csv(results_path, index=False)
test_predictions.to_csv(predictions_path, index=False)

print('\nValidation summary (selection metrics):')
print(validation_df.to_string(index=False))

print('\nTest metrics by model and modality:')
print(results_df.to_string(index=False))

print('\nSaved benchmark outputs:')
print(' -', validation_path)
print(' -', results_path)
print(' -', predictions_path)
