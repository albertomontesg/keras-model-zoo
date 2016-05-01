from distutils.core import setup

setup(
    name='Keras Model Zoo',
    version='0.0.1',
    author='Alberto Montes',
    author_email='alm59321@gmail.com',
    maintainer='Alberto Montes',
    maintainer_email='alm59321@gmail.com',
    packages=['kerasmodelzoo'],
    package_data = {'kerasmodelzoo': ['data/*.npy']},
    license='MIT',
    description="Keras Model Zoo",
    install_requires=['keras', 'progressbar2', 'six'],
)
