from setuptools import setup


setup(
    name='traf',
    version='1.0',
    py_modules=['traf'],
    install_requires=[
        'Click',
        'numpy>=1.19.2',
    ],
    tests_requires=[
        'pytest',
    ],
    entry_points='''
        [console_scripts]
        traf=traf.main:main
    '''
)
