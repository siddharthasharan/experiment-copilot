"""Experiment Copilot: an AI that reviews A/B tests — then gets graded on whether
it catches the traps."""

__version__ = "0.1.0"

from .data import GeneratedExperiment, Scenario, demo_scenarios, generate_experiment
from .stats import analyze_experiment, review_summary, sample_size_proportions

__all__ = [
    "GeneratedExperiment",
    "Scenario",
    "analyze_experiment",
    "demo_scenarios",
    "generate_experiment",
    "review_summary",
    "sample_size_proportions",
]
