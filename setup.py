from setuptools import setup, find_packages

setup(
    name='facebox',
    version='0.1.0',
    description='Real-time face detection with Haar cascades',
    author='Your Name',
    packages=find_packages(),
    install_requires=[
        'opencv-python'
    ],
    python_requires='>=3.8',
)
