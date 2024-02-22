import os
import platform
import sys

import pkg_resources
from setuptools import find_packages, setup

requirements = []
if sys.platform.startswith("linux") and platform.machine() == "x86_64":
    requirements.append("triton==2.0.0")
setup(
    name="peft-ser",
    py_modules=["peft_ser"],
    version="0.0.4",
    description="Parameter Efficient Fine-tuning on Speech Emotion Recognition.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    readme="README.md",
    python_requires=">=3.8",
    author="Tiantian Feng",
    license="BSD",
    packages=find_packages(exclude=["tests*"]),
    install_requires=requirements
    + [
        str(r)
        for r in pkg_resources.parse_requirements(
            open(os.path.join(os.path.dirname(__file__), "requirements.txt"))
        )
    ],
    entry_points={
        "console_scripts": ["whisper=whisper.transcribe:cli"],
    },
    include_package_data=True,
    extras_require={"dev": ["pytest", "scipy", "black", "flake8", "isort"]},
)
