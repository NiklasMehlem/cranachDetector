from setuptools import setup, find_packages

setup(
    name="cranach_detector",
    version="1.0.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "opencv-python",
        "dlib",
        "matplotlib",
        "numpy",
        "insightface",
        "mtcnn",
        "Pillow",
    ],
    author="Niklas Mehlem",
    description="GUI + Gesichtserkennung via dlib/MTCNN/RetinaFace",
    python_requires=">=3.7",
)
