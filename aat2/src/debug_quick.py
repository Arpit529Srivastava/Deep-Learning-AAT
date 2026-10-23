import numpy as np, tensorflow as tf
from tensorflow.keras import layers, models
s=np.load("aat2/eyes/synth_eyes_64.npz"); x=s["x"][...,None].astype("float32")/255.; y=s["y"].astype("float32"); face=s["face"]
faces=np.unique(face); rng=np.random.default_rng(42); rng.shuffle(faces); n=len(faces)
val_f=set(faces[int(.15*n):int(.30*n)]); inv=np.array([f in val_f for f in face])
tr=np.random.default_rng(1).choice(np.where(~inv)[0],12000,replace=False); va=np.where(inv)[0]
def cb(f): return [layers.Conv2D(f,3,padding="same",use_bias=False),layers.BatchNormalization(),layers.ReLU(),layers.MaxPooling2D()]
m=models.Sequential([layers.Input((64,64,1))]+cb(32)+cb(64)+cb(128)+[layers.GlobalAveragePooling2D(),layers.Dense(1,activation="sigmoid")])
m.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
m.fit(x[tr],y[tr],batch_size=128,epochs=2,verbose=2,validation_data=(x[va],y[va]))
p=m.predict(x[va],verbose=0).ravel(); print("val mean pred", p.mean(), "acc", ((p>.5)==y[va]).mean())
# BN in training mode on val
pt=np.concatenate([m(x[va][i:i+256],training=True).numpy().ravel() for i in range(0,len(va),256)]); print("val acc with batch stats", ((pt>.5)==y[va]).mean())
rs=np.random.default_rng(0).choice(tr,2000,replace=False)
pi=m.predict(x[rs],verbose=0).ravel(); print("TRAIN inference-mode acc", ((pi>.5)==y[rs]).mean(), "mean pred", pi.mean(), "label mean", y[rs].mean())
bn=[l for l in m.layers if isinstance(l,layers.BatchNormalization)][0]
print("bn moving mean", bn.moving_mean.numpy()[:5], "moving var", bn.moving_variance.numpy()[:5])
xb=x[rs][:256]; c=m.layers[0](xb).numpy(); print("conv1 batch mean", c.mean((0,1,2))[:5], "var", c.var((0,1,2))[:5])
