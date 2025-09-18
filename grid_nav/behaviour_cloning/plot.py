#!/usr/bin/env python3

"""
Streamlit app for visualizing training metrics from a completed training run.
Run with: streamlit run plot.py -- --metrics <path_to_metrics.json>
"""

import argparse
import json
from pathlib import Path

import streamlit as st


def load_metrics(metrics_file):
    """Load metrics from JSON file."""
    with open(metrics_file, 'r') as f:
        return json.load(f)


def main():
    st.set_page_config(
        page_title="Training Loss Plot",
        page_icon="📊"
    )

    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--metrics', type=str, required=True,
                        help='Path to metrics JSON file')

    try:
        args = parser.parse_args()
        metrics_file = Path(args.metrics)

        if not metrics_file.exists():
            st.error(f"Metrics file not found: {metrics_file}")
            return

    except SystemExit:
        st.error("Please provide --metrics argument")
        st.code("streamlit run plot.py -- --metrics path/to/metrics.json")
        return

    # Load metrics
    try:
        metrics = load_metrics(metrics_file)
    except Exception as e:
        st.error(f"Error loading metrics: {e}")
        return

    # Title
    st.title("Training and Test Loss")

    # Extract data
    epochs = [m['epoch'] for m in metrics]
    train_losses = [m['train_loss'] for m in metrics]
    test_losses = [m['test_loss'] for m in metrics]

    # Prepare data for line chart
    chart_data = {
        'Epoch': epochs,
        'Train Loss': train_losses,
        'Test Loss': test_losses
    }

    # Display the chart
    st.line_chart(chart_data, x='Epoch', y=['Train Loss', 'Test Loss'],
                  use_container_width=True, height=500)


if __name__ == '__main__':
    main()