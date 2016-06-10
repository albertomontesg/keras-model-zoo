from setuptools import setup

setup(
    name='KerasModelZoo',
    description='A package with models for Keras',
    version='0.0.1',
    author='Alberto Montes',
    author_email='alm59321@gmail.com',
    maintainer='Alberto Montes',
    maintainer_email='alm59321@gmail.com',
    download_url = 'https://github.com/albertomontesg/kerasmodelzoo/tarball/0.0.1',
    packages=['kerasmodelzoo'],
    package_data = {'kerasmodelzoo': ['data/*.npy']},
    keywords=['Keras', 'deeplearning', 'Theano', 'Tensorflow', 'models'],
    license='MIT',
    install_requires=['keras', 'progressbar2', 'six'],
)
