import numpy as np
import pandas as pd
from iesseeg.metrics import paired_auc,auc_influence,summarize

def test_identical_predictions_have_zero_paired_uncertainty():
    f=pd.DataFrame({'patient_id':np.repeat(range(50),2),'fold':np.repeat(np.arange(50)//10,2),'label':np.repeat(np.arange(50)%2,2),'recording_id':np.repeat(np.arange(50).astype(str),2),'crop_id':[0,1]*50,'probability':np.random.default_rng(42).random(100)})
    result=paired_auc(f,f.sample(frac=1,random_state=7))
    assert result=={'delta_auroc':0.,'ci_low':0.,'ci_high':0.}
    influence=auc_influence(f)
    np.testing.assert_allclose(np.sqrt(np.mean(influence.to_numpy()**2)/50),summarize(f)['cv_auroc_se'])
