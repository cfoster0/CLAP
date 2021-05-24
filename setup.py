from setuptools import setup, find_packages

setup(
    name="clap-jax",
    packages=find_packages(),
    version="0.0.1",
    license="MIT",
    description="CLAP - Contrastive Language-Audio Pretraining",
    author="Charles Foster",
    author_email="",
    url="https://github.com/cfoster0/CLAP",
    keywords=[
        "artificial intelligence",
        "deep learning",
        "contrastive learning",
        "audio",
    ],
    install_requires=[
        "click",
        "click-option-group",
        "einops>=0.3",
        "flax",
        "hydra",
        "jax",
        "jaxlib",
        "lm_dataformat",
        "optax",
        "torch",
        "torchaudio",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
    ],
)
