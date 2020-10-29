from setuptools import setup


setup(
    name='corelib',
    version='1.0',
    py_modules=['corelib'],
    python_requires=">=3.9",
    install_requires=[
        'numpy>=1.19.2',
        'marshmallow>=3.8.0',
    ],
    tests_requires=[
        'pytest',
    ]
)
