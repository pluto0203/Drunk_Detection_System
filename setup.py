"""Setup script for the Drunk Detection System package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="drunk-detection-system",
    version="1.0.0",
    author="Pluto",
    description="AI-powered drunk driver detection using MobileNetV3 and MQ3 sensor fusion",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "tensorflow>=2.13.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "opencv-python>=4.8.0",
        "mediapipe>=0.10.0",
        "pandas>=2.0.0",
        "Pillow>=10.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "pyyaml>=6.0",
        "python-dotenv>=1.0.0",
        "mlflow>=2.8.0",
    ],
    extras_require={
        "edge": ["tflite-runtime>=2.13.0"],
        "dashboard": ["flask>=3.0.0"],
        "notifications": ["python-telegram-bot>=20.0"],
        "advanced": [
            "tensorflow-model-optimization>=0.7.0",
            "optuna>=3.4.0",
        ],
        "dev": ["ruff>=0.1.0", "mypy>=1.7.0", "pytest>=7.0.0"],
    },
    entry_points={
        "console_scripts": [
            "drunk-train=scripts.train:main",
            "drunk-evaluate=scripts.evaluate:main",
            "drunk-export=scripts.export_tflite:main",
            "drunk-distill=scripts.distill:main",
            "drunk-qat=scripts.qat_export:main",
            "drunk-tune=scripts.tune:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
