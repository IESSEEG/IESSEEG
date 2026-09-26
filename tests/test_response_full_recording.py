import numpy as np
import pandas as pd
import pytest
from response_full_recording import window_chunks, check_protocol

@pytest.mark.parametrize('seconds',[4,16,30])
@pytest.mark.parametrize('duration',[1801.,71372.,172801.])
def test_every_complete_window_once(seconds,duration):
    chunks=window_chunks(duration,seconds,32)
    indices=[j for first,n in chunks for j in range(first,first+n)]
    assert indices==list(range(int(duration//seconds)))
    assert 0<=duration-len(indices)*seconds<seconds
    # A partial inference minibatch must retain the same weight per window.
    p=np.linspace(0.,1.,len(indices))**2
    streamed=sum(p[first:first+n].sum() for first,n in chunks)/len(indices)
    np.testing.assert_allclose(streamed,p.mean())

def test_checkpoint_patient_partitions_must_match():
    frame=pd.DataFrame({'patient_id':range(50),'fold':np.repeat(range(5),10)})
    protocol=dict(task='sustained',condition='PRE',test_fold=0,train_patients=list(range(20,50)),validation_patients=list(range(10,20)),test_patients=list(range(10)))
    check_protocol(protocol,frame,0,'PRE')
    protocol['train_patients'][0]=0
    with pytest.raises(ValueError,match='train_patients'):check_protocol(protocol,frame,0,'PRE')

def test_streaming_groups_do_not_mix_recordings(tmp_path,monkeypatch):
    import torch
    from types import SimpleNamespace
    import response_full_recording as run
    monkeypatch.setattr(run,'read_interval',lambda path,start,duration:np.broadcast_to(int(path.name)+start/30+np.arange(duration*200)//6000,(21,duration*200)))
    monkeypatch.setattr(run,'prepare_windows',lambda raw,model,seconds:raw[0,::seconds*200,None].astype('float32'))
    class Model:
        def encode(self,x):return x
        def head(self,h):return h
    a=SimpleNamespace(ckpts=tmp_path,model='stub',branch='finetuned',visit='PRE',inference_batch_size=2,workers=0,device=torch.device('cpu'))
    rows=[dict(patient_id=i,recording_id=str(i),source=str(i),source_duration=91,label=i,fold=0) for i in (0,1)]
    run.infer_many(a,Model(),rows,30,fixed_fold=0)
    for row in rows:
        with np.load(run.predpath(a,'finetuned',0,row)) as z:
            expected=torch.tensor([row['patient_id']+i for i in range(3)],dtype=torch.float32).sigmoid().numpy()
            np.testing.assert_allclose(z['probability'],expected)
            assert len(z['probability'])==3
    # Resuming a complete recording does not rerun it.
    monkeypatch.setattr(run,'read_interval',lambda *args:pytest.fail('Unexpected reread'))
    run.infer_many(a,Model(),rows,30,fixed_fold=0)
