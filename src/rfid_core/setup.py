from setuptools import setup


setup(
    name='rfid_core',
    version='1.0',
    py_modules=['rfid_core'],
    install_requires=[
        'numpy>=1.19.2',
    ],
    tests_requires=[
        'pytest',
    ],
)
