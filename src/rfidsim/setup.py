from setuptools import setup


setup(
    name='rfidsim',
    version='1.0',
    py_modules=['rfidsim'],
    install_requires=[
        'Click',
        'numpy>=1.19.2',
    ],
    tests_requires=[
        'pytest',
    ],
    entry_points='''
        [console_scripts]
        rfidsim=rfidsim.main:main
    '''
)
