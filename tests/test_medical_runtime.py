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
