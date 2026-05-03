import setuptools

setuptools.setup(
    name='jarvis-local',
    version='0.1.1',
    package_dir={"": "."},
    packages=setuptools.find_packages(),
    python_requires=">=3.9",
    author="Timo Hueser",
    author_email="jarvismocap@gmail.com",
    url="git@github.com:JohnsonLabJanelia/JARVIS-HybridNet.git",
    description="A small example package",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "opencv-python",
        "matplotlib",
        "tqdm",
        "yacs",
        "ruamel.yaml",
        "imgaug",
        "tensorboard",
        "ipywidgets",
        "joblib",
        "pandas",
        "seaborn",
        "Click",
        "streamlit",
        "streamlit_option_menu",
        "inquirer",
        "altair",
    ],
    entry_points={
        'console_scripts': [
            'jarvis-local = jarvis.ui.jarvis:cli',
        ],
    }
)
