# SPAR

The main entry and supporting files of SPAR are available in this
repository. The core source-reliability scoring and training modules are not
included while the paper is under review. The complete implementation will be
released after the paper is accepted.

## Usage

### Installation

The project uses Python 3.9 and PyTorch.

```bash
git clone https://github.com/YXS649/SPAR.git
cd SPAR
python -m pip install -r requirements.txt
```

### Data

Prepare the SEED or SEED-IV DE-LDS feature files as MATLAB `.mat` files and
organize the three sessions as:

```text
feature_root/
├── 1/
├── 2/
└── 3/
```

### Training

SEED:

```bash
CUDA_VISIBLE_DEVICES=0 bash run.sh \
  --dataset seed3 \
  --data-dir /path/to/SEED/features \
  --output-dir outputs/seed3_seed20 \
  --seed 20
```

SEED-IV:

```bash
CUDA_VISIBLE_DEVICES=0 bash run.sh \
  --dataset seed4 \
  --data-dir /path/to/SEED-IV/features \
  --output-dir outputs/seed4_seed20 \
  --seed 20
```

The current public version cannot run end to end until the withheld core
modules are released.
