from setuptools import setup


setup(
    name='rfidsim',
    version='1.0',
    py_modules=['rfidsim'],
    python_requires=">=3.8",
    install_requires=[
        'Click',
        'numpy>=1.19.2',
        'marshmallow>=3.8.0',
    ],
    tests_requires=[
        'pytest',
    ],
    entry_points='''
        [console_scripts]
        rfidsim=rfidsim.main:cli
    '''
)
