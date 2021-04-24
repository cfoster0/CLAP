# CLAP
Contrastive Language-Audio Pretraining

In due time this repo will be full of lovely things, I hope.

Feel free to check out the Issues if you're interested in contributing. Leave a note saying what interests you. :)

```python
from jax import random
from clap.clap import CLAP

key1, key2 = random.split(random.PRNGKey(0), 2)

text = random.randint(key1, (2, 16,), 0, 10000)
audio = random.uniform(key1, (2, 8, 512))

model = CLAP(
    text_num_tokens = 10000,
    text_dim = 512,
    text_depth = 6,
    text_heads = 8,
    audio_dim = 512,
    audio_depth = 6,
    audio_heads = 8
)

params = model.init(key2, text, audio)

sim = model.apply(params, text, audio) # (2, 2)
```
