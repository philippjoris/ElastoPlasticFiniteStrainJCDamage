from setuptools_scm import get_version
from skbuild import setup

project_name = "GMatElastoPlasticFiniteStrainSimo"

setup(
    name=project_name,
    description="Simo elasto-plastic model (finite strain mechanics).",
    long_description="Simo elasto-plastic model (finite strain mechanics).",
    version=get_version(),
    license="MIT",
    author="Tom de Geus",
    author_email="tom@geus.me",
    url=f"https://github.com/tdegeus/{project_name}",
    packages=[f"{project_name}"],
    package_dir={"": "python"},
    cmake_install_dir=f"python/{project_name}"
)
