import setuptools

setuptools.setup(
    name='jarvis-local',
    version='0.1.1',
    package_dir={"": "."},
    packages=setuptools.find_packages(),
    python_requires=">=3.6",
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
        "ruamel.yaml==0.17.40",
        "imgaug==0.4.0",
        "tensorboard",
        "ipywidgets",
        "joblib",
        "pandas",
        "seaborn",
        "Click",
        "streamlit==1.11.0",
        "streamlit_option_menu",
        "inquirer==2.8.0",
        "altair==5" 
    ],
    entry_points={
        'console_scripts': [
            'jarvis-local = jarvis.ui.jarvis:cli',
        ],
    }
)
