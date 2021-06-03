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
audio = random.uniform(key1, (2, 1024, 80))

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

params = model.init(key2, text, audio)
loss = model.apply(params, text, audio)

# after a lot of training

sim = model.apply(params, text, audio, return_loss = False) # (2, 2)
```

Use Hydra's config system to swap out dataset and model configurations

```
python preprocess.py +preprocessing/dataset=commonvoice
python train.py +model/audio=vit +model/text=transformer +optimizer=standard +training=standard
```


## Citations

[OpenAI blog post "CLIP: Connecting Text and Images"](https://openai.com/blog/clip/)

```bibtex
@article{radford2021learning,
  title={Learning transferable visual models from natural language supervision},
  author={Radford, Alec and Kim, Jong Wook and Hallacy, Chris and Ramesh, Aditya and Goh, Gabriel and Agarwal, Sandhini and Sastry, Girish and Askell, Amanda and Mishkin, Pamela and Clark, Jack and others},
  journal={arXiv preprint arXiv:2103.00020},
  year={2021}
}
```

```bibtex
@article{jia2021scaling,
  title={Scaling Up Visual and Vision-Language Representation Learning With Noisy Text Supervision},
  author={Jia, Chao and Yang, Yinfei and Xia, Ye and Chen, Yi-Ting and Parekh, Zarana and Pham, Hieu and Le, Quoc V and Sung, Yunhsuan and Li, Zhen and Duerig, Tom},
  journal={arXiv preprint arXiv:2102.05918},
  year={2021}
}
```

Much of the code behind the various transformer configurations has been adapted from Niccolò Zanichelli's [repository of Flax vision transformer modules](https://github.com/NZ99/self-attention-experiments-vision).

Citation block courtesy of MicPie's awesome parallel project ["Contrastive Language-Aminoacid Sequence Pretraining"](https://github.com/MicPie/clasp).
