from setuptools import setup


setup(
    name='wisim',
    version='1.0',
    py_modules=['wisim'],
    install_requires=[
        'Click',
        'numpy>=1.19.2',
    ],
    tests_requires=[
        'pytest',
    ],
    entry_points='''
        [console_scripts]
        wisim=wisim.main:main
    '''
)
