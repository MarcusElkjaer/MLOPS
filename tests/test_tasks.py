import pytest
import os
from unittest.mock import MagicMock, call
from invoke import Context
from tasks import (
    create_environment,
    requirements,
    dev_requirements,
    preprocess_data,
    train,
    run_test,
    docker_build,
    build_docs,
    serve_docs,
)

WINDOWS = os.name == "nt"

@pytest.fixture
def mock_context():
    """Fixture to provide a mocked Context."""
    ctx = MagicMock(spec=Context)  # Mock the Context class
    return ctx


def test_create_environment(mock_context):
    """Test create_environment task."""
    create_environment(mock_context)
    mock_context.run.assert_called_once_with(
        "conda create --name reddit_forecast python=3.11 pip --no-default-packages --yes",
        echo=True,
        pty=not WINDOWS,  # Match implementation
    )


def test_requirements(mock_context):
    """Test requirements task."""
    requirements(mock_context)
    expected_calls = [
        call("pip install -U pip setuptools wheel", echo=True, pty=not WINDOWS),
        call("pip install -r requirements.txt", echo=True, pty=not WINDOWS),
        call("pip install -e .", echo=True, pty=not WINDOWS),
    ]
    mock_context.run.assert_has_calls(expected_calls)


def test_dev_requirements(mock_context):
    """Test dev_requirements task."""
    # Manually invoke the `requirements` task before calling `dev_requirements`
    requirements(mock_context)
    dev_requirements(mock_context)
    
    expected_calls = [
        call("pip install -U pip setuptools wheel", echo=True, pty=not WINDOWS),
        call("pip install -r requirements.txt", echo=True, pty=not WINDOWS),
        call("pip install -e .", echo=True, pty=not WINDOWS),
        call('pip install -e .["dev"]', echo=True, pty=not WINDOWS),
    ]
    mock_context.run.assert_has_calls(expected_calls, any_order=False)


def test_preprocess_data(mock_context):
    """Test preprocess_data task."""
    preprocess_data(mock_context)
    mock_context.run.assert_called_once_with(
        "python src/reddit_forecast/data.py data/raw data/processed",
        echo=True,
        pty=not WINDOWS,  # Match implementation
    )


def test_train(mock_context):
    """Test train task."""
    train(mock_context)
    mock_context.run.assert_called_once_with(
        "python src/reddit_forecast/train.py",
        echo=True,
        pty=not WINDOWS,  # Match implementation
    )


def test_run_test(mock_context):
    """Test test task."""
    run_test(mock_context)
    expected_calls = [
        call("PYTHONPATH=src coverage run -m pytest", echo=True, pty=not WINDOWS),
        call("coverage report", echo=True, pty=not WINDOWS),
    ]
    mock_context.run.assert_has_calls(expected_calls)


def test_docker_build(mock_context):
    """Test docker_build task."""
    docker_build(mock_context, progress="plain")
    expected_calls = [
        call(
            "docker build -t train:latest . -f dockerfiles/train.dockerfile --progress=plain",
            echo=True,
            pty=not WINDOWS,
        ),
        call(
            "docker build -t api:latest . -f dockerfiles/api.dockerfile --progress=plain",
            echo=True,
            pty=not WINDOWS,
        ),
    ]
    mock_context.run.assert_has_calls(expected_calls)


def test_build_docs(mock_context):
    """Test build_docs task."""
    build_docs(mock_context)
    mock_context.run.assert_called_once_with(
        "mkdocs build --config-file docs/mkdocs.yaml --site-dir build",
        echo=True,
        pty=not WINDOWS,  # Match implementation
    )


def test_serve_docs(mock_context):
    """Test serve_docs task."""
    serve_docs(mock_context)
    mock_context.run.assert_called_once_with(
        "mkdocs serve --config-file docs/mkdocs.yaml",
        echo=True,
        pty=not WINDOWS,  # Match implementation
    )
