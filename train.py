#!/usr/bin/env python3
"""Run only RankEqualRDS-v1 on SEED / SEED-IV (transductive LOSO)."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from rank_equal_rds import data
from rank_equal_rds.model import RankEqualNet
from rank_equal_rds.trainer import fit
from rank_equal_rds.metrics import classification_metrics_from_probabilities

PROTOCOL = 'RANK_EQUAL_RDS_V1'

DEFAULT_DATA_DIRS = {
    # Home-relative defaults keep the repository portable while allowing the
    # current machine's canonical SEED location to work without extra flags.
    'seed3': [
        Path.home() / 'dataset' / 'SEED' / 'ExtractedFeatures_by_session',
        Path.home() / 'dataset' / 'seed',
    ],
    'seed4': [
        Path.home() / 'dataset' / 'SEED_IV' / 'eeg_feature_smooth',
        Path.home() / 'EEG' / 'SEED_IV' / 'eeg_feature_smooth',
    ],
}


def resolve_data_dir(dataset, explicit=None):
    """Resolve a compatible MAT feature root without embedding a user path."""
    if explicit is not None:
        candidate = Path(explicit).expanduser()
        if not candidate.is_dir():
            raise ValueError('The requested data directory does not exist: {}'.format(candidate))
        return candidate
    env_name = f'STABLE_RDS_FINAL_{dataset.upper()}_PATH'
    env_value = os.environ.get(env_name)
    if env_value:
        candidate = Path(env_value).expanduser()
        if not candidate.is_dir():
            raise ValueError('{} points to a missing directory: {}'.format(env_name, candidate))
        return candidate
    for candidate in DEFAULT_DATA_DIRS[dataset]:
        if candidate.is_dir():
            return candidate
    choices = ', '.join(str(item) for item in DEFAULT_DATA_DIRS[dataset])
    raise ValueError(
        'No default {} feature directory was found. Provide --data-dir or {}. '
        'Expected a 3-session/15-subject MAT root containing de_LDS arrays; '
        'searched: {}'.format(dataset, env_name, choices)
    )


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def make_loaders(features, labels, metadata, target_id, batch_size):
    source_ids = [i for i in range(15) if i != target_id]
    def loader(subject, target=False, evaluate=False):
        x = features[subject]
        if len(x) < 2:
            raise ValueError('BatchNorm training requires at least two samples per domain')
        y = np.zeros((len(x), 1), dtype=np.int64) if target else labels[subject]
        ds = data.CustomDataset(
            x, y, return_index=evaluate,
            trial_ids=metadata[subject]['trial_ids'], window_ids=metadata[subject]['window_ids'],
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=not evaluate,
                          drop_last=(not evaluate and len(x) >= batch_size))
    return ([loader(i) for i in source_ids], loader(target_id, True),
            loader(target_id, True, True), source_ids)


def run_fold(features, labels, metadata, session, subject, args, device):
    started = time.perf_counter()
    sources, target_train, target_eval, source_ids = make_loaders(
        features, labels, metadata, subject, args.batch_size)
    model = RankEqualNet(number_of_category=3 if args.dataset == 'seed3' else 4).to(device)
    iterations = math.ceil(args.epochs * (3394 if args.dataset == 'seed3' else 820) / args.batch_size)
    result = fit(
        model, sources, target_train, target_eval, source_ids, iterations=iterations,
        diagnostic_seed=args.seed + 700000 + session * 10000 + subject * 100,
        device=device, eval_interval=args.eval_interval,
    )
    # Only after fit completes do real labels enter metric computation.
    target_labels = labels[subject].reshape(-1)
    final = classification_metrics_from_probabilities(result['final_probabilities'], target_labels)
    last3 = classification_metrics_from_probabilities(result['probabilities'], target_labels)
    folder = args.output_dir / f'session{session}_subject{subject}'
    folder.mkdir()
    np.savez_compressed(
        folder / 'predictions.npz', labels=target_labels,
        sample_ids=np.arange(len(target_labels)),
        trial_ids=metadata[subject]['trial_ids'], window_ids=metadata[subject]['window_ids'],
        probabilities=result['probabilities'], final_probabilities=result['final_probabilities'],
        predictions=result['probabilities'].argmax(axis=1),
        last_k_probabilities=result['last_k_probabilities'], last_k_iterations=result['last_k_iterations'],
        active_source_ids=np.asarray(result['active_source_ids']),
    )
    torch.save(model.state_dict(), folder / 'final_model.pt')
    write_json(folder / 'selection.json', result['selection'])
    row = dict(
        protocol_id=PROTOCOL, dataset=args.dataset, seed=args.seed, session_id=session, subject_id=subject,
        iterations=iterations, active_source_ids=result['active_source_ids'],
        weights=[0.25] * 4, weight_search=False, proxy_training=False,
        last_k_iterations=result['last_k_iterations'].tolist(),
        periodic_labeled_evaluations=0, post_training_labeled_evaluations=1,
        checkpoint_selection='chronological_last3_no_labels',
        final={k: v for k, v in final.items() if k != 'predictions'},
        last3={k: v for k, v in last3.items() if k != 'predictions'},
        elapsed_seconds=time.perf_counter() - started,
    )
    write_json(folder / 'metrics.json', row)
    print(f'Completed session={session} subject={subject}: Last-3 ACC={last3["ensemble_acc"]:.4f}', flush=True)
    return row


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['seed3', 'seed4'], required=True)
    parser.add_argument('--data-dir', type=Path, help='Local feature directory (never uploaded)')
    parser.add_argument('--output-dir', type=Path, required=True, help='New output directory; existing paths are refused')
    parser.add_argument('--seed', type=int, default=20)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--eval-interval', type=int, default=500, help='Unlabeled probability snapshot interval')
    parser.add_argument('--sessions', type=int, nargs='+', default=[0, 1, 2], help='Zero-based session IDs')
    parser.add_argument('--subjects', type=int, nargs='+', default=list(range(15)), help='Zero-based target IDs')
    parser.add_argument('--device', default='auto', help='auto, cpu, cuda:0, ...')
    parser.add_argument('--check-data', action='store_true', help='Validate data without training')
    args = parser.parse_args(argv)
    try:
        args.data_dir = resolve_data_dir(args.dataset, args.data_dir)
    except ValueError as exc:
        parser.error(str(exc))
    if args.output_dir.exists():
        parser.error('Refusing to overwrite an existing output directory')
    if args.batch_size < 2 or args.epochs < 1 or args.eval_interval < 1 or args.seed < 0:
        parser.error('Require batch-size >= 2, epochs/eval-interval > 0, seed >= 0')
    for values, count in ((args.sessions, 3), (args.subjects, 15)):
        if len(set(values)) != len(values) or any(i < 0 or i >= count for i in values):
            parser.error('Session/subject IDs must be distinct and within the dataset range')
    return args


def main(argv=None):
    args = parse_args(argv)
    device = torch.device(('cuda:0' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device)
    random.seed(0)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    data.dataset_path[args.dataset] = str(args.data_dir)
    features, labels, metadata = data.load_data(args.dataset, preserve_dtype=True, return_metadata=True)
    if len(features) != 3 or any(len(session) != 15 for session in features):
        raise ValueError('Expected three sessions with fifteen subjects each')
    for session in features:
        for i, x in enumerate(session):
            if not np.isfinite(x).all():
                raise ValueError('Non-finite input features')
            session[i] = data.norminy(x.copy())
    args.output_dir.mkdir(parents=True)
    config = {
        'protocol_id': PROTOCOL,
        'dataset': args.dataset,
        'run': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        'versions': {'torch': torch.__version__, 'numpy': np.__version__},
    }
    write_json(args.output_dir / 'run_config.json', config)
    if args.check_data:
        write_json(args.output_dir / 'data_check.json', [
            dict(session_id=s, subject_id=i, shape=list(x.shape))
            for s, session in enumerate(features) for i, x in enumerate(session)])
        print('DATA_CHECK_PASSED')
        return
    rows = []
    # Continuous RNG across folds, in historical session/subject order.
    for session in sorted(args.sessions):
        for subject in sorted(args.subjects):
            rows.append(run_fold(features[session], labels[session], metadata[session], session, subject, args, device))
            write_json(args.output_dir / 'session_summary.json', rows)
    summary = {'protocol_id': PROTOCOL, 'folds': len(rows), 'aggregation': 'unweighted_fold_mean_and_population_std'}
    for stage in ('final', 'last3'):
        summary[stage] = {metric: {'mean': float(np.mean([r[stage][metric] for r in rows])),
                                  'std': float(np.std([r[stage][metric] for r in rows]))}
                          for metric in ('ensemble_acc', 'macro_f1', 'balanced_acc')}
    write_json(args.output_dir / 'overall_summary.json', summary)
    with (args.output_dir / 'metrics_summary.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['session_id', 'subject_id', 'final_acc_percent', 'last3_acc_percent', 'last3_macro_f1', 'last3_bacc'])
        for row in rows:
            writer.writerow([row['session_id'], row['subject_id'], row['final']['ensemble_acc'],
                             row['last3']['ensemble_acc'], row['last3']['macro_f1'], row['last3']['balanced_acc']])
    write_json(args.output_dir / 'run_complete.json', {'status': 'complete', 'protocol_id': PROTOCOL, 'folds': len(rows)})


if __name__ == '__main__':
    main()
