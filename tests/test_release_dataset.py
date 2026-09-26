"""Public manifest joins must use explicit participant IDs, never row position."""
import json
import numpy as np
import pandas as pd
import pytest
from iesseeg.dataset import Dataset
from iesseeg.tabular import _masks


def test_patient_mapping_mismatch_is_rejected(tmp_path):
    participants=pd.DataFrame({'participant_id':[f'sub-{i}' for i in range(100)],'patient_id':range(100)})
    participants.sample(frac=1,random_state=2).to_csv(tmp_path/'participants.tsv',sep='\t',index=False)
    rows=[]
    for subset,n in [('original_long',150),('clinician_selected',600),('simulated_routine',200)]:
        for i in range(n):rows.append(dict(recording_id=f'{subset}-{i}',source_id=str(i),subset=subset,patient_id=i%100,participant_id=f'sub-{i%100}',source_recording_id=f'long-{i%100}',path='unused.edf'))
    data=pd.DataFrame(rows);data.to_csv(tmp_path/'recordings.tsv',sep='\t',index=False)
    (tmp_path/'splits').mkdir()
    for task,n in [('diagnosis',100),('response',50)]:
        pd.DataFrame({'patient_id':range(n),'fold':np.arange(n)%5}).to_csv(tmp_path/f'splits/{task}_folds.csv',index=False)
    assert Dataset(tmp_path).validate(False)['participants']==100
    data.loc[0,'participant_id']='sub-99';data.to_csv(tmp_path/'recordings.tsv',sep='\t',index=False)
    with pytest.raises(ValueError,match='mappings disagree'):Dataset(tmp_path).validate(False)


def test_repeated_segments_follow_their_patient_partition():
    frame=pd.DataFrame({'patient_id':np.repeat(range(50),10),'fold':np.repeat(np.arange(50)%5,10)})
    for f in range(5):
        tr,va,te=_masks(frame,f)
        assert (tr.sum(),va.sum(),te.sum())==(300,100,100)
    frame.loc[0,'fold']=1
    with pytest.raises(ValueError,match='Patient overlap'):_masks(frame,0)
