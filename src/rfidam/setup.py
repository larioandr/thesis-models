from setuptools import setup


setup(
    name='rfidam',
    version='1.0',
    py_modules=['rfidam'],
    install_requires=[
        'Click',
        'numpy>=1.19.2',
    ],
    tests_requires=[
        'pytest',
    ],
    entry_points='''
        [console_scripts]
        rfidam=rfidam.main:main
    '''
)
