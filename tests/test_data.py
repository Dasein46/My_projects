from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from PIL import Image
from cv_common.data import validate_images, patient_split, load_manifest


def test_group_split_preserves_patients_and_seed():
    frame = pd.DataFrame({'filename': [f'{i}.png' for i in range(40)],
                          'patient_id': [str(i // 2) for i in range(40)],
                          'label': ['a', 'b'] * 20})
    parts = patient_split(frame)
    again = patient_split(frame)
    assert set().union(*(set(p.filename) for p in parts.values())) == set(frame.filename)
    for name, part in parts.items():
        assert part.equals(again[name])
        for other, second in parts.items():
            if name != other:
                assert set(part.patient_id).isdisjoint(second.patient_id)


@pytest.mark.parametrize('column', ['patient_id', 'filename', 'pixel_sha256'])
def test_official_split_rejects_overlap(column):
    frame = pd.DataFrame({'filename': ['a','b','c'], 'patient_id': ['1','2','3'],
                          'pixel_sha256':['x','y','z'], 'split':['train','validation','test']})
    frame.loc[1, column] = frame.loc[0, column]
    with pytest.raises(ValueError, match='overlaps'):
        patient_split(frame)


def test_missing_patient_cannot_fallback_to_images():
    with pytest.raises(ValueError, match='patient_id'):
        patient_split(pd.DataFrame({'filename':['a']}))


def test_duplicate_pixels_different_names(tmp_path):
    im = Image.new('RGB', (8,8), 'red')
    im.save(tmp_path/'a.png'); im.save(tmp_path/'b.bmp')
    with pytest.raises(ValueError, match='Repeated'):
        validate_images(pd.DataFrame({'filename':['a.png','b.bmp']}), tmp_path)


def test_high_bit_depth_is_not_silently_clipped(tmp_path):
    Image.fromarray(np.full((8,8), 4095, dtype=np.uint16)).save(tmp_path/'a.tiff')
    with pytest.raises(ValueError, match='intensity'):
        validate_images(pd.DataFrame({'filename':['a.tiff']}), tmp_path)


def test_official_split_preserved(tmp_path):
    rows=[]
    for i, split in enumerate(['train','validation','test']):
        Image.new('RGB',(8,8),(i,20,30)).save(tmp_path/f'{i}.png')
        rows.append({'filename':f'{i}.png','patient_id':str(i),'label':'a','split':split})
    pd.DataFrame(rows).to_csv(tmp_path/'manifest.csv', index=False)
    parts=load_manifest(tmp_path/'manifest.csv',tmp_path)
    assert [parts[s].patient_id.iloc[0] for s in parts] == ['0','1','2']


def test_unseen_training_class_rejected():
    frame=pd.DataFrame({'filename':['a','b','c'], 'patient_id':['1','2','3'],
                        'label':['a','b','a'], 'split':['train','validation','test']})
    with pytest.raises(ValueError, match='Classes missing'):
        patient_split(frame)
