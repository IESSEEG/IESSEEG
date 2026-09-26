import pytest
from iesseeg_paper.splits import load_fold_manifest, validate_patient_folds


@pytest.mark.parametrize("task,n", [("diagnosis", 100), ("immediate", 50), ("sustained", 50)])
def test_frozen_partitions_cover_each_patient_once(task, n):
    frame = load_fold_manifest(task)
    assert len(frame) == n
    assert frame.groupby("fold").size().eq(n // 5).all()
    validate_patient_folds(frame, task, require_complete=True)


def test_repeated_pre_post_views_inherit_folds_and_drift_is_rejected():
    frame = load_fold_manifest("sustained")
    views = frame.loc[frame.index.repeat(8)].reset_index(drop=True)
    validate_patient_folds(views, "sustained", require_complete=True)
    views.loc[0, "fold"] = (views.loc[0, "fold"] + 1) % 5
    with pytest.raises(ValueError, match="differ"):
        validate_patient_folds(views, "sustained")


def test_missing_or_unknown_patients_are_rejected():
    frame = load_fold_manifest("diagnosis")
    with pytest.raises(ValueError, match="Incomplete"):
        validate_patient_folds(frame.iloc[:-1], "diagnosis", require_complete=True)
    frame.loc[0, "patient_id"] = -1
    with pytest.raises(ValueError, match="differ"):
        validate_patient_folds(frame, "diagnosis")
