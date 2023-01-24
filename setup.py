from setuptools import setup

setup(
    name='pme',
    version='0.1.0',
    description='A package to train and generate activity-level '
                'embeddings for process mining',
    url='https://github.com/Kookaburra99/pme',
    author='Pedro Gamallo Fernández',
    author_email='pedro.gamallo.fernandez@usc.es',
    license='GNU General Public License',
    packages=['pme'],
    install_requires=['pandas',
                      'scikit-learn',
                      'pm4py',
                      'torch',
                      'gensim',
                      'karateclub'],
    classifiers=[
        'Programming Language :: Python :: 3'
    ],
)
