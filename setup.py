import setuptools

with open('requirements.txt', 'r') as f:
    install_requires = f.read().splitlines()

setuptools.setup(name='grav_waldo',
                 version='0.1.0',
                 author='Tibério A. Pereira',
                 description='waveform anomaly detector',
                 packages=['grav_waldo'],
                 install_requires=install_requires,
                 classifiers=["Programming Language :: Python :: 3",
                              "Operating System :: OS Independent"])

