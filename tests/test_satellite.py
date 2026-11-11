import importlib.util
import sys
from pathlib import Path
import numpy as np
import pytest
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

spec=importlib.util.spec_from_file_location('satellite',Path(__file__).parents[1]/'Generation of satellite images/train.py')
sat=importlib.util.module_from_spec(spec);sys.modules[spec.name]=sat;spec.loader.exec_module(sat)
torch.set_num_threads(1)


def make_pairs(root):
    for i in range(5):
        for season, offset in [('summer',0),('winter',15)]:
            Image.new('RGB',(101,79),(i*40+offset,80,90)).save(root/f'city_{i}_{season}.jpg')
    return sat.pair_cities(root)


def test_city_split_and_all_border_pixels(tmp_path):
    pairs=make_pairs(tmp_path)
    train,val,test=sat.split_cities(pairs,.2,66)
    assert set(train).isdisjoint(val) and set(train).isdisjoint(test) and set(val).isdisjoint(test)
    assert len(train)+len(val)+len(test)==len(pairs)
    tiles=sat.build_tiles(val,3,4)
    coverage=np.zeros((79,101),dtype=int)
    for tile in tiles:
        a,b,c,d=tile.box;coverage[b:d,a:c]+=1
    assert (coverage==1).all()
    x,y=sat.SeasonalTiles(tiles,64)[0]
    assert x.shape==y.shape==(3,64,64)
    assert 0<=x.min()<=x.max()<=1


def test_missing_pair(tmp_path):
    Image.new('RGB',(8,8)).save(tmp_path/'foo_summer.jpg')
    with pytest.raises(ValueError, match='counterpart'):
        sat.pair_cities(tmp_path)


def test_wrong_shape_rejected(tmp_path):
    pairs=make_pairs(tmp_path)
    Image.new('RGB',(17,13)).save(pairs['city_0'][1])
    with pytest.raises(ValueError, match='sizes'):
        sat.build_tiles(pairs,2,2)


def test_metrics_compare_target_and_weight_partial_batch():
    x=torch.zeros(3,3,4,4); y=torch.ones_like(x);y[-1]=.5
    mae,mse=sat.evaluate(nn.Identity(),DataLoader(TensorDataset(x,y),batch_size=2),torch.device('cpu'))
    assert mae==pytest.approx(2.5/3)
    assert mse==pytest.approx(2.25/3)


def test_real_unet_forward_backward_without_download():
    model=sat.SeasonalTranslator(pretrained=False)
    x=torch.rand(2,3,64,64);target=torch.rand_like(x)
    prediction=model(x)
    assert prediction.shape==target.shape
    loss=nn.MSELoss()(prediction,target);loss.backward()
    assert torch.isfinite(loss)
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.parameters())


def test_exact_city_duplicate_rejected(tmp_path):
    pairs=make_pairs(tmp_path)
    pairs['copy']=pairs['city_0']
    with pytest.raises(ValueError, match='different cities'):
        sat.validate_distinct_cities(pairs)


def test_cli_training_checkpoint_and_test(tmp_path):
    import json
    data=tmp_path/'data';data.mkdir();make_pairs(data)
    output=tmp_path/'run'
    sat.main(['--data-dir',str(data),'--output-dir',str(output),'--epochs','1',
              '--rows','1','--cols','2','--image-size','64','--batch-size','2','--evaluate-test'])
    report=json.loads((output/'metrics.json').read_text())
    assert report['test'] is not None
    assert np.isfinite(report['test']['mse'])
    assert (output/'best_model.pt').exists()
    assert Image.open(output/'validation_example.png').size==(192,64)
