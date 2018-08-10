import sys

from setuptools import find_packages, setup


def main():
    install_requires = [
        'click',
        'colorama; platform_system=="Windows"'
    ]

    setup_requires = ['setuptools_scm']
    if {'pytest', 'test', 'ptr'}.intersection(sys.argv):
        setup_requires.append('pytest-runner')

    tests_require = ['pytest']

    setup(
        name='tt2bf',
        description='TruthTable to (minimal) BooleanFormula converter',
        url='https://github.com/Lipen/tt2bf',
        author='Konstantin Chukharev',
        author_email='lipen00@gmail.com',
        license='GNU GPLv3',
        python_requires='>=3.6',
        package_dir={'': 'src'},
        packages=find_packages('src'),
        use_scm_version={
            'write_to': 'src/tt2bf/version.py',
            'version_scheme': 'post-release',
            # 'local_scheme': lambda _: '',
            'local_scheme': 'dirty-tag',
        },
        install_requires=install_requires,
        setup_requires=setup_requires,
        tests_require=tests_require,
        entry_points={
            'console_scripts': [
                'tt2bf = tt2bf:cli',
            ]
        },
        zip_safe=False,
    )


if __name__ == '__main__':
    main()
