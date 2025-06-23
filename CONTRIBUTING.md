# Contributing to quicfire-tools

We welcome contributions to quicfire-tools! This document outlines how to contribute to the project.

## Reporting Issues

If you encounter bugs or have feature requests, please submit an issue on the [GitHub issues page](https://github.com/silvxlabs/quicfire-tools/issues).

When reporting issues, please include:
- A clear description of the problem
- Steps to reproduce the issue
- Your operating system and Python version
- Relevant error messages or output

## Development Setup

1. Fork the repository and clone your fork:
   ```bash
   git clone https://github.com/your-username/quicfire-tools.git
   cd quicfire-tools
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the package in development mode with test dependencies:
   ```bash
   pip install -e .
   pip install -r requirements/test_requirements.txt
   ```

4. Install pre-commit hooks for code formatting and linting:
   ```bash
   pip install pre-commit
   pre-commit install
   ```

5. Run the tests to ensure everything is working:
   ```bash
   pytest
   ```

## Making Changes

1. Create a new branch for your changes:
   ```bash
   git checkout -b your-feature-branch
   ```

2. Make your changes, following the existing code style and conventions.

3. Add or update tests as needed. All new functionality should include tests.

4. Run the test suite to ensure your changes don't break existing functionality:
   ```bash
   pytest
   ```

5. Commit your changes with a descriptive commit message. Pre-commit hooks will automatically format and lint your code:
   ```bash
   git add .
   git commit -m "Add feature: brief description of changes"
   ```

   Note: If pre-commit hooks make changes to your files, you'll need to add and commit those changes as well.

6. Push your branch and submit a pull request:
   ```bash
   git push origin your-feature-branch
   ```

## Code Style and Linting

This project uses several tools to maintain code quality:

- **Black**: Automatic code formatting
- **Flake8**: Code linting and style checking
- **Pre-commit hooks**: Automatic formatting and linting on commit

### Running Code Quality Checks

Pre-commit hooks will automatically run when you commit, but you can also run them manually:

```bash
# Run all pre-commit hooks on all files
pre-commit run --all-files

# Run black formatting
black .

# Run flake8 linting
flake8 .
```

### Code Style Guidelines

- Code is automatically formatted with Black - don't worry about manual formatting
- Follow existing naming conventions and patterns in the codebase
- Write clear, concise code with appropriate docstrings
- Keep functions and classes focused and well-documented
- Use type hints where appropriate

## Testing

This project uses pytest for testing. Run the full test suite with:
```bash
pytest
```

Tests are located in the `tests/` directory and should cover new functionality and edge cases.

## Release Process

Understanding how releases work can help you see how your contributions make it into the published package:

1. **Pull Request Review**: All changes go through pull request review on the main branch
2. **Automated Testing**: Every PR triggers automated tests across multiple Python versions (3.9-3.12) and operating systems (Ubuntu, Windows)
3. **Code Quality Checks**: Pre-commit hooks ensure code formatting and linting standards
4. **Release Creation**: Maintainers create GitHub releases when ready to publish
5. **Automated Publishing**: When a release is published, an automated workflow:
   - First publishes to Test PyPI
   - Tests the package installation from Test PyPI
   - Then publishes to the main PyPI
   - Finally validates the PyPI installation

This ensures that every release is thoroughly tested before reaching end users. Contributors don't need to worry about the release process - just focus on making quality contributions that pass the automated checks.

## Getting Help

If you need help with development or have questions about contributing:
- Open an issue on the [GitHub issues page](https://github.com/silvxlabs/quicfire-tools/issues)
- Check the [documentation](https://silvxlabs.github.io/quicfire-tools/)
- Review existing issues and pull requests for similar questions

## License

By contributing to quicfire-tools, you agree that your contributions will be licensed under the same MIT license as the project.