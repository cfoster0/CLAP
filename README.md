# CLAP

Contrastive Language-Audio Pretraining

In due time this repo will be full of lovely things, I hope.

Feel free to check out the Issues if you're interested in contributing. Leave a note saying what interests you. :)

## Requirements

```bash
$ python setup.py install --user
```

You'll also need to use the latest Optax (>=0.0.7)

```bash
$ pip install git+git://github.com/deepmind/optax.git
```

## Use

```python
from jax import random, numpy as np
from clap.models import CLAP

key1, key2 = random.split(random.PRNGKey(0), 2)

text = random.randint(key1, (2, 16,), 0, 256)
audio = random.uniform(key1, (2, 8, 512))

text_mask = np.ones((2, 16), dtype = bool)
audio_mask = np.ones((2, 8), dtype = bool)

text_config = {
    'kind': 'transformer',
    'depth': 8,
    'dim': 512,
    'heads': 8,
    'vocab': 256,
    'projection_dim': 512,
    'rotary_qk': True,
}

audio_config = {
    'kind': 'vit', 
    'depth': 8, 
    'dim': 512, 
    'heads': 8, 
    'patch_shape': [4, 80], 
    'projection_dim': 512, 
    'rotary_qk': True,
}

model = CLAP(
    text_config = text_config,
    audio_config = audio_config,
)

params = model.init(key2, text, audio, text_mask, audio_mask)
loss = model.apply(params, text, audio, text_mask, audio_mask)

# after a lot of training

sim = model.apply(params, text, audio, text_mask, audio_mask, return_loss = False) # (2, 2)
```

Use Hydra's config system to swap out model configurations

```
python train.py +model/audio=vit +model/text=transformer +optimizer=standard +training=standard
```
