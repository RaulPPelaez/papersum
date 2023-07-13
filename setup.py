from setuptools import setup, find_packages

setup(
    name='papersum',
    version='0.1.0',    # Update this for new versions
    packages=find_packages(),
    install_requires=[
        line.strip() for line in open('requirements.txt', 'r').readlines()
    ],
    entry_points={
        'console_scripts': [
            'papersum=papersum.papersum:run',  # if you have a main function in papersum.py
        ],
    },
    author='RaulPPelaez',  # Update this
    author_email='raulppelaez@gmail.com',  # Update this
    description='An LLM paper summarization tool',  # Update this
)
