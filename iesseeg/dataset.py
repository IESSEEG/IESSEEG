"""Read the release tables and materialize a compatibility workspace.

Only symbolic links are created for EEG files; original downloads are untouched.
The adapter preserves the released patient mapping and experimental row order.
"""
from pathlib import Path
from importlib.resources import files
import json
import numpy as np
import pandas as pd


class Dataset:
    def __init__(self, root):
        self.root = Path(root).expanduser().resolve()
        self.participants = pd.read_csv(self.root / 'participants.tsv', sep='\t')
        self.recordings = pd.read_csv(self.root / 'recordings.tsv', sep='\t', dtype={'source_id': str})
        if self.participants.patient_id.duplicated().any() or self.recordings.recording_id.duplicated().any():
            raise ValueError('Duplicate release identifiers')
        self.folds = {task: pd.read_csv(self.root / f'splits/{task}_folds.csv') for task in ('diagnosis', 'response')}

    def validate(self, require_edf=True):
        p, r = self.participants, self.recordings
        expected = {'original_long': 150, 'clinician_selected': 600, 'simulated_routine': 200}
        if len(p) != 100 or r.groupby('subset').size().to_dict() != expected:
            raise ValueError('The complete v1.1 metadata are required')
        link = r.patient_id.map(p.set_index('patient_id').participant_id)
        if not link.eq(r.participant_id).all():
            raise ValueError('Participant and recording mappings disagree')
        if not r.groupby('source_recording_id').patient_id.nunique().eq(1).all():
            raise ValueError('A source recording maps to multiple participants')
        for task, f in self.folds.items():
            if f.patient_id.duplicated().any() or set(f.fold) != set(range(5)):
                raise ValueError(f'Invalid {task} fold manifest')
        if require_edf:
            missing = [path for path in r.path if not (self.root / path).is_file()]
            if missing:
                raise FileNotFoundError(f'{len(missing)} EDF files missing, including {missing[0]}')
        return {'participants': len(p), 'recordings': expected, 'edf_files_checked': require_edf}

    def ordered(self, subset):
        # Preserve the paper runners' input order independently of release table order.
        order = json.loads(files('iesseeg').joinpath('recording_order.json').read_text())
        ids = order[subset]
        return self.recordings.set_index('recording_id').loc[ids].reset_index()


def link_file(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink() and target.resolve() == source.resolve():
        return
    if target.exists() or target.is_symlink():
        raise FileExistsError(f'Refusing to replace a different file: {target}')
    target.symlink_to(source.resolve())


def prepare(dataset, work):
    ds = Dataset(dataset)
    ds.validate(require_edf=False)
    work = Path(work).expanduser().resolve()
    workspace = work / 'workspace'
    data = workspace / 'data'
    (data / 'raw_data').mkdir(parents=True, exist_ok=True)
    participants = ds.participants.set_index('patient_id')

    def legacy(r):
        d = pd.DataFrame({
            'short_recording_id': r.source_id, 'long_recording_id': r.source_long_id,
            'case_control_label': np.where(r.patient_diagnosis_label.eq(1), 'CASE', 'CONTROL'),
            'pre_post_treatment_label': r.visit.replace({'CONTROL': 'PRE'}),
            'sleep_awake_label': r.sleep_awake_label.fillna('UNKNOWN').str.upper(),
            'patient_id': r.patient_id})
        for new, old in [('AgeAtEEG1y', 'age_at_eeg_years'), ('AOOmo', 'age_at_onset_months'), ('LeadtimeUKISS', 'lead_time_ukiss')]:
            d[new] = r.patient_id.map(participants[old])
        for new, old in [('meaningful_responder', 'treatment_response_label'), ('immediate_responder', 'initial_response_label')]:
            d[new] = r.patient_id.map(participants[old]).map({0: 'Non-responder', 1: 'Responder'}).fillna('UNKNOWN')
        d['human_label'] = r.clinician_diagnosis_label
        return d

    frames = {}
    for name, subset, filename in [('selected','clinician_selected','final_short_merged.csv'), ('routine','simulated_routine','final_test.csv')]:
        rows = ds.ordered(name)
        frames[name] = legacy(rows)
        frames[name].to_csv(data / filename, index=False)
        folder = data / 'raw_data' / ('edf' if name == 'selected' else 'test_edf')
        for r in rows.itertuples():
            if (ds.root / r.path).is_file():
                link_file(ds.root / r.path, folder / f'{r.source_id}.edf')
    manifest = ds.recordings.copy()
    manifest['source'] = manifest.path.map(lambda x: str(ds.root / x))
    manifest['source_duration'] = manifest.duration_seconds
    manifest['recording_id'] = manifest.source_id
    manifest['pre_post_treatment_label'] = manifest.visit.replace({'CONTROL':'PRE'})
    manifest['case_control_label'] = np.where(manifest.patient_diagnosis_label.eq(1), 'CASE', 'CONTROL')
    manifest['kind'] = manifest.subset.map({'original_long':'long', 'clinician_selected':'smith_development', 'simulated_routine':'smith_evaluation'})
    manifest['label'] = manifest.treatment_response_label
    manifest['fold'] = manifest.patient_id.map(ds.folds['response'].set_index('patient_id').fold)
    manifest.to_csv(work / 'inputs.csv', index=False)
    for r in ds.recordings[ds.recordings.subset.eq('original_long')].itertuples():
        if (ds.root / r.path).is_file():
            link_file(ds.root / r.path, workspace / 'long_eeg' / f'{r.source_id}.edf')
    lookup = ds.recordings.set_index('recording_id')
    for path in (ds.root / 'sampling').glob('*.csv'):
        c = pd.read_csv(path)
        c['recording_id'] = c.source_recording_id.map(lookup.source_id)
        c['source'] = c.source_recording_id.map(lookup.path).map(lambda x: str(ds.root / x))
        c['source_duration'] = c.source_recording_id.map(lookup.duration_seconds)
        c['label'] = c.patient_id.map(participants.treatment_response_label).astype(int)
        c['sample_id'] = [f'{v}_{rid}_{crop:02d}' for v,rid,crop in zip(c.pre_post_treatment_label,c.recording_id,c.crop_id)]
        (work / 'sampling').mkdir(exist_ok=True)
        c.to_csv(work / 'sampling' / path.name, index=False)
    # Exact ordered outer partitions used by the diagnostic training runners.
    spec = pd.read_csv(ds.root / 'splits/diagnosis_recordings.tsv', sep='\t')
    for (fold, split), group in spec.groupby(['fold','split']):
        rows = lookup.loc[group.sort_values('row_order').recording_id].reset_index()
        dest = work / 'splits/case_control' / f'fold_{fold}' / f'{split}.csv'
        dest.parent.mkdir(parents=True, exist_ok=True)
        legacy(rows).to_csv(dest, index=False)
    (work / 'dataset.json').write_text(json.dumps({'dataset':str(ds.root),'workspace':str(workspace)},indent=2)+'\n')
    return workspace
