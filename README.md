# CLAP

Contrastive Language-Audio Pretraining

In due time this repo will be full of lovely things, I hope.

Feel free to check out the Issues if you're interested in contributing. Leave a note saying what interests you. :)

## Requirements

```bash
$ python setup.py install --user
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

model = CLAP(
    text_vocab = 256,
    text_dim = 512,
    text_depth = 6,
    text_heads = 8,
    audio_dim = 512,
    audio_depth = 6,
    audio_heads = 8
)

params = model.init(key2, text, audio, text_mask, audio_mask)
loss = model.apply(params, text, audio, text_mask, audio_mask)

# after a lot of training

sim = model.apply(params, text, audio, text_mask, audio_mask, return_loss = False) # (2, 2)
```
