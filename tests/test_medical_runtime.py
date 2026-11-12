"""Synthetic integration checks; pretrained downloads are disabled."""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','3')
os.environ.setdefault('TF_NUM_INTEROP_THREADS','1')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','1')
import json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import pytest
from cv_common.data import keras_generators


def test_vgg_preprocessing_and_non_augmented_validation(tmp_path):
    from tensorflow.keras.applications.vgg16 import preprocess_input
    rows=[]
    for i in range(3):
        path=tmp_path/f'{i}.png'
        Image.new('RGB',(224,224),(10+i,80,120)).save(path)
        rows.append({'filename':str(path),'label':'a' if i%2 else 'b'})
    parts={s:pd.DataFrame(rows) for s in ['train','validation','test']}
    gens=keras_generators(parts,['a','b'],batch_size=2,vgg=True,augment={'rotation_range':30})
    x,y=gens['validation'][0]
    expected=preprocess_input(np.array(Image.open(rows[0]['filename']),dtype=np.float32))
    np.testing.assert_allclose(x[0],expected)
    np.testing.assert_array_equal(x,gens['validation'][0][0])
    assert len(gens['validation'][1][0])==1


@pytest.mark.parametrize('folder',['Classification of body parts in X-Rays','Classification of organs in MRI','Classification of organs in US','Classification of fetal organs in US'])
def test_original_architecture_one_training_batch(folder):
    import tensorflow as tf
    tf.keras.backend.clear_session()
    path=next((Path(__file__).parents[1]/folder).glob('*.ipynb'))
    cells=json.loads(path.read_text())['cells']
    source=next(''.join(c['source']) for c in cells if 'model = Sequential()' in ''.join(c['source']))
    source=source.replace("weights='imagenet'","weights=None")
    scope={'tf':tf,'Sequential':tf.keras.Sequential}
    scope.update({name:getattr(tf.keras.layers,name) for name in ['Dense','Flatten','Conv2D','MaxPooling2D']})
    exec(source,scope)
    model=scope['model'];n=model.output_shape[-1]
    x=np.random.default_rng(0).random((2,224,224,3)).astype('float32')
    y=tf.keras.utils.to_categorical([0,1],n)
    result=model.train_on_batch(x,y)
    assert np.isfinite(result).all()
    assert model(x,training=False).shape==(2,n)


def test_fastai_multilabel_pipeline(tmp_path):
    import torch
    from fastai.vision.all import ImageDataLoaders, MultiCategoryBlock, Resize, vision_learner, resnet50
    torch.set_num_threads(1)
    rows=[]
    for i in range(8):
        path=tmp_path/f'{i}.png';Image.new('RGB',(64,64),(i,10,100)).save(path)
        rows.append({'filename':str(path),'labels':['a','b'] if i%2 else ['a'],'is_valid':i>=6})
    df=pd.DataFrame(rows)
    from cv_common.multilabel import build_loaders
    dls=build_loaders(df,['a','b'],image_size=64,batch_size=2,augment=False)
    learn=vision_learner(dls,resnet50,pretrained=False)
    x,y=dls.one_batch();loss=learn.loss_func(learn.model(x),y);loss.backward()
    assert torch.isfinite(loss)
    test_dl=dls.test_dl(df.iloc[:3],with_labels=True,shuffle=False)
    probs,targets=learn.get_preds(dl=test_dl)
    assert probs.shape==targets.shape==(3,2)
    assert ((probs>=0)&(probs<=1)).all()


def test_keras_fit_checkpoint_and_complete_evaluation(tmp_path):
    import tensorflow as tf
    from cv_common.data import evaluate_classifier
    rows=[]
    for i in range(3):
        path=tmp_path/f'eval_{i}.png';Image.new('RGB',(224,224),(i,10,20)).save(path)
        rows.append({'filename':str(path),'label':['a','b','a'][i]})
    parts={s:pd.DataFrame(rows) for s in ['train','validation','test']}
    gens=keras_generators(parts,['a','b'],batch_size=2)
    model=tf.keras.Sequential([tf.keras.Input(shape=(224,224,3)),tf.keras.layers.GlobalAveragePooling2D(),tf.keras.layers.Dense(2,activation='softmax')])
    model.compile(optimizer='adam',loss='categorical_crossentropy')
    checkpoint=tmp_path/'best.keras'
    model.fit(gens['train'],validation_data=gens['validation'],epochs=1,verbose=0,
              callbacks=[tf.keras.callbacks.ModelCheckpoint(checkpoint,save_best_only=True)])
    restored=tf.keras.models.load_model(checkpoint)
    report=evaluate_classifier(restored,gens['test'])
    assert np.array(report['confusion_matrix']).sum()==3


@pytest.mark.parametrize('folder', ['Classification of body parts in X-Rays',
    'Classification of organs in MRI', 'Classification of organs in US', 'Classification of fetal organs in US'])
def test_medical_notebook_cells_on_synthetic_manifest(tmp_path, monkeypatch, folder):
    import re
    import matplotlib
    matplotlib.use('Agg')
    root=Path(__file__).parents[1]
    cells=json.loads(next((root/folder).glob('*.ipynb')).read_text())['cells']
    scope={};exec(''.join(cells[2]['source']),scope)
    data=tmp_path/'data';data.mkdir();rows=[]
    for split in ['train','validation','test']:
        for label in scope['CLASSES']:
            i=len(rows);name=f'{i}.png'
            Image.fromarray(np.random.default_rng(i).integers(0,256,(32,32,3),dtype=np.uint8)).save(data/name)
            rows.append(dict(filename=name,label=label,patient_id=f'p{i}',split=split))
    pd.DataFrame(rows).to_csv(data/'manifest.csv',index=False)
    output=tmp_path/'run'
    monkeypatch.chdir(root);monkeypatch.setenv('DATA_ROOT',str(data));monkeypatch.setenv('OUTPUT_DIR',str(output))
    monkeypatch.delenv('MANIFEST_PATH',raising=False);monkeypatch.delenv('ALLOW_FILENAME_PATIENT_IDS',raising=False)
    for cell in cells:
        if cell['cell_type']=='code':
            source=''.join(cell['source']).replace("weights='imagenet'",'weights=None')
            exec(re.sub(r'epochs=\d+','epochs=1',source),scope)
    report=json.loads((output/'test_metrics.json').read_text())
    assert np.array(report['confusion_matrix']).sum()==len(scope['CLASSES'])
    assert sum(scope['classify_image'](data/'0.png').values())==pytest.approx(1,abs=1e-5)
    assert not any(Path(x).is_absolute() for x in pd.read_csv(output/'split_manifest.csv').filename)
    high=data/'high.tiff';Image.fromarray(np.full((8,8),4095,dtype=np.uint16)).save(high)
    with pytest.raises(ValueError,match='intensity'):scope['classify_image'](high)


def test_fastai_small_training_split_fit_export_reload(tmp_path):
    import torch
    from fastai.vision.all import vision_learner,resnet50,load_learner
    from fastai.callback.tracker import SaveModelCallback
    from cv_common.multilabel import build_loaders
    from cv_common.data import read_rgb
    torch.set_num_threads(1);rows=[]
    for i in range(7):
        path=tmp_path/f'{i}.png';Image.new('RGB',(64,64),(i*20,50,100)).save(path)
        rows.append(dict(filename=str(path),labels=['a','b'] if i%2 else ['a'],is_valid=i>=5))
    frame=pd.DataFrame(rows)
    dls=build_loaders(frame,['a','b'],image_size=64,batch_size=2,augment=False)
    sizes=[len(x) for x,_ in dls.train];assert sum(sizes)==5 and min(sizes)>=2
    learn=vision_learner(dls,resnet50,pretrained=False,model_dir=tmp_path)
    learn.fine_tune(1,base_lr=1e-4,cbs=[SaveModelCallback(monitor='valid_loss',fname='best',reset_on_fit=False)])
    learn.export(tmp_path/'model.pkl')
    restored=load_learner(tmp_path/'model.pkl',cpu=True)
    probabilities=restored.predict(read_rgb(frame.filename.iloc[0]))[2]
    assert probabilities.shape==(2,) and torch.isfinite(probabilities).all()


@pytest.mark.parametrize('count,batch',[(2,32),(3,2),(7,2),(13,2),(33,32)])
def test_fastai_batch_boundaries(tmp_path,count,batch):
    from cv_common.multilabel import build_loaders
    rows=[]
    for i in range(count+2):
        p=tmp_path/f'{i}.png';Image.new('RGB',(64,64),(i,20,50)).save(p)
        rows.append(dict(filename=str(p),labels=['a'],is_valid=i>=count))
    dls=build_loaders(pd.DataFrame(rows),['a'],image_size=64,batch_size=batch,augment=False)
    sizes=[len(x) for x,_ in dls.train];assert sum(sizes)==count and min(sizes)>=2


def test_multilabel_notebook_manifest_and_checksums(tmp_path,monkeypatch):
    from cv_common.data import validate_images,patient_split,save_split_manifest
    cells=json.loads(next((Path(__file__).parents[1]/'Classification of organs in X-Rays (2)').glob('*.ipynb')).read_text())['cells']
    (tmp_path/'train').mkdir();out=tmp_path/'run';out.mkdir();rows=[];manifest=[]
    for i,split in enumerate(['train','train','validation','test']):
        name=f'train/{i}-c.png';Image.new('RGB',(64,64),(i*20,30,70)).save(tmp_path/name)
        rows.append(dict(SOPInstanceUID=str(i),Target=' '.join(map(str,range(22)))))
        manifest.append(dict(filename=name,patient_id=f'p{i}',split=split))
    pd.DataFrame(rows).to_csv(tmp_path/'train_df.csv',index=False)
    pd.DataFrame(manifest).to_csv(tmp_path/'manifest.csv',index=False)
    monkeypatch.delenv('MANIFEST_PATH',raising=False)
    scope=dict(pd=pd,os=os,DATA_ROOT=tmp_path,OUTPUT_DIR=out,SEED=12345,validate_images=validate_images,patient_split=patient_split,save_split_manifest=save_split_manifest)
    exec(''.join(cells[2]['source']),scope);exec(''.join(cells[3]['source']),scope)
    assert scope['parts']['test'].patient_id.tolist()==['p3']
    with pytest.raises(ValueError,match='Unknown'):scope['decode_target']('22')
    monkeypatch.setenv('MANIFEST_PATH',str(out/'split_manifest.csv'))
    Image.new('RGB',(64,64),(200,200,200)).save(tmp_path/manifest[-1]['filename'])
    with pytest.raises(ValueError,match='differ'):exec(''.join(cells[3]['source']),scope)
    monkeypatch.delenv('MANIFEST_PATH');pd.DataFrame(manifest[:-1]).to_csv(tmp_path/'manifest.csv',index=False)
    with pytest.raises(ValueError,match='patient_id'):exec(''.join(cells[3]['source']),scope)


def test_classifier_rejects_wrong_probability_shape():
    from cv_common.data import evaluate_classifier
    from types import SimpleNamespace
    gen=SimpleNamespace(shuffle=False,n=2,class_indices={'a':0,'b':1},reset=lambda:None)
    model=SimpleNamespace(predict=lambda *a,**kw:np.ones((2,1)))
    with pytest.raises(ValueError):evaluate_classifier(model,gen)
