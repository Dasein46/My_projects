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


def test_run_directory_protects_existing_results(tmp_path):
    from cv_common.data import create_run_directory
    p=create_run_directory(tmp_path/'run');(p/'model').write_text('original')
    with pytest.raises(ValueError,match='empty'):create_run_directory(p)
    assert (p/'model').read_text()=='original'


def test_inference_rejects_multiple_frames(tmp_path):
    from cv_common.data import read_rgb
    p=tmp_path/'multi.tiff';Image.new('L',(8,8)).save(p,save_all=True,append_images=[Image.new('L',(8,8),10)])
    with pytest.raises(ValueError,match='Multi-frame'):read_rgb(p)


def test_portable_manifest_and_changed_pixels(tmp_path):
    from cv_common.data import save_split_manifest
    import shutil
    root=tmp_path/'data';root.mkdir();rows=[]
    for i,split in enumerate(['train','validation','test']):
        Image.new('RGB',(8,8),(i,20,30)).save(root/f'{i}.png')
        rows.append(dict(filename=f'{i}.png',patient_id=str(i),label='a',split=split))
    pd.DataFrame(rows).to_csv(root/'manifest.csv',index=False)
    parts=load_manifest(root/'manifest.csv',root)
    save_split_manifest(parts,root,root/'split_manifest.csv')
    moved=tmp_path/'moved';shutil.move(root,moved)
    assert len(load_manifest(moved/'split_manifest.csv',moved))==3
    Image.new('RGB',(8,8),(200,20,30)).save(moved/'0.png')
    with pytest.raises(ValueError,match='differ'):load_manifest(moved/'split_manifest.csv',moved)


def test_path_escape_rejected(tmp_path):
    p=tmp_path/'outside.png';Image.new('RGB',(8,8)).save(p)
    root=tmp_path/'root';root.mkdir()
    with pytest.raises(ValueError,match='within'):validate_images(pd.DataFrame({'filename':['../outside.png']}),root)


def test_inference_rejects_high_bit_depth(tmp_path):
    from cv_common.data import read_rgb
    p=tmp_path/'high.tiff';Image.fromarray(np.full((8,8),4095,dtype=np.uint16)).save(p)
    with pytest.raises(ValueError,match='intensity'):read_rgb(p)
