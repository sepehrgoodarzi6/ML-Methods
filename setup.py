from setuptools import setup, find_packages

setup(
    name='automated_machineLearning_methods',
    version='0.1.0',
    description="This package is a comprehensive Python toolkit designed to streamline and simplify the application of various machine learning models for data analytics and prediction tasks. Featuring a broad range of algorithms from classic logistic regression, K-nearest neighbors, and decision trees to advanced neural networks, the package caters to both traditional techniques and state-of-the-art methods. Built with user-friendliness in mind, ML-Methods bridges the gap between data science theory and practical application, enabling users to deploy sophisticated machine learning workflows with ease.",
    author='Sepehr Goodarzi',
    author_email='sepehrgoodarzi6@gmail.com',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'tensorflow',
        'matplotlib',
        'sklearn',
        'seaborn',
        'statsmodels',
        'scikit-learn',
        'numpy',
        'keras'
    ]
)