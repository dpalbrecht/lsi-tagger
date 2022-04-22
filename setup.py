from setuptools import setup, find_packages


setup(
    name='lsi_tagger',
    version='0.1',
    license='MIT',
    author='David Albrecht',
    author_email='davidpabloalbrecht@gmail.com',
    description='LSI based, pairwise tag extraction intended for e-commerce product descriptions.',
    packages=find_packages(),
    url='https://github.com/dpalbrecht/lsi-tagger',
    keywords=['nlp','e-commerce','keyword-extraction','latent-semantic-indexing'],
    install_requires=['gensim','numpy','nltk']
)