from setuptools import setup


def readme():
    with open('README.md') as f:
        return f.read()


setup(name='pyqumo',
      version='0.1.4',
      description='Queueing Models in Python',
      long_description=readme(),
      classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.7',
        'Intended Audience :: Science/Research',
      ],
      keywords='queueing systems, markov chains',
      url='https://github.com/larioandr/pyqumo',
      author='Andrey Larionov',
      author_email='larioandr@gmail.com',
      license='MIT',
      packages=['pyqumo'],
      py_modules=['pyqumo'],
      scripts=[],
      python_requires=">=3.8",
      install_requires=[
          'numpy',
          'scipy',
          'tabulate',
      ],
      include_package_data=True,
      zip_safe=False,
      setup_requires=["pytest-runner"],
      tests_require=["pytest"], 
    )
