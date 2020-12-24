from setuptools import setup


def readme():
    with open('README.md') as f:
        return f.read()


setup(name='pycsmaca',
      version='0.1.2',
      description='Wireless networks models',
      long_description=readme(),
      classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.7',
        'Topic :: Communications',
        'Intended Audience :: Science/Research',
      ],
      keywords='pydesim, csma/ca, wireless networks',
      url='https://github.com/larioandr/pycsmaca',
      author='Andrey Larionov',
      author_email='larioandr@gmail.com',
      license='MIT',
      packages=['pycsmaca'],
      py_modules=['pycsmaca'],
      scripts=[],
      install_requires=[
          'scipy',
          'pydesim',
          'pyqumo',
      ],
      dependency_links=[
          'git+https://github.com/larioandr/pydesim.git#egg=pydesim',
          'git+https://github.com/larioandr/pyqumo.git#egg=pyqumo',
      ],
      include_package_data=True,
      zip_safe=False,
      setup_requires=["pytest-runner", "pytest-repeat"],
      tests_require=["pytest", 'pyqumo', 'pydesim'],
    )
