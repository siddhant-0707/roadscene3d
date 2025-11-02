from setuptools import setup, find_packages

setup(
    name="roadscene3d",
    version="0.1.0",
    description="RoadScene3D: A Self-Supervised 3D Scene Understanding & Flywheel Pipeline",
    author="Siddhant Chauhan",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "open3d>=0.17.0",
        "mlflow>=2.7.0",
        "tqdm>=4.65.0",
    ],
)
