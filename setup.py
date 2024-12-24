from setuptools import setup, find_packages

setup(
    name='PoseEstimation',
    version='0.1.0',
    author='Marc Hartley',
    author_email='marc.hartley@hotmail.com',
    description='A Python package for pose estimation with skeleton tracking features, object detection (YoloV4) and ArUco detection',
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url='http://github.com/marchartley/PoseEstimation',
    packages=find_packages(),
    install_requires=[
        'scipy',
        'oepncv-contrib-python',
        'numpy'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    keywords='pose estimation, skeleton tracking, aruco, object detection',
    python_requires='>=3.6',
)
