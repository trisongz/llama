import sys
from pathlib import Path
from setuptools import setup, find_packages

if sys.version_info.major != 3:
    raise RuntimeError("This package requires Python 3+")

pkg_name = 'llama'
gitrepo = 'trisongz/llama'

root = Path(__file__).parent
version = '0.0.1'

def get_requirements(name: str = None):
    if name is None: name = 'requirements'
    if 'requirements' not in name: name = f'requirements.{name}'
    if not name.endswith('.txt'): name = f'{name}.txt'
    return [line.strip() for line in root.joinpath(name).read_text().splitlines() if ('#' not in line[:5] and line.strip())]


requirements = get_requirements()


if sys.version_info.minor < 8:
    requirements.append('typing_extensions')

extras = {
    'int8': get_requirements('int8'),
}

args = {
    'packages': find_packages(include = [f'{pkg_name}', f'{pkg_name}.*']),
    'install_requires': requirements,
    'include_package_data': True,
    'long_description': root.joinpath('README.md').read_text(encoding='utf-8'),
    'entry_points': {
        "console_scripts": [
            # "configz = configz.cli:run_cmd",
        ]
    },
    'extras_require': extras,
}

setup(
    name = pkg_name,
    version = version,
    url = f'https://github.com/{gitrepo}',
    license = 'GNU General Public License',
    description='Inference code for LLaMA models',
    author='Meta Platforms, Inc. and affiliates',
    author_email='ts@growthengineai.com',
    long_description_content_type="text/markdown",
    classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License',
        'Programming Language :: Python :: 3.7',
        'Topic :: Software Development :: Libraries',
    ],
    **args
)


