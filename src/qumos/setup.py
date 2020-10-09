from setuptools import setup


setup(
    name='qumos',
    version='1.0',
    py_modules=['qumos'],
    install_requires=[
        'Click',
        'numpy>=1.19.2',
    ],
    tests_requires=[
        'pytest',
    ],
    entry_points='''
        [console_scripts]
        qumos=qumos.main:main
    '''
)
