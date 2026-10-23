"""
tests/test_network.py
=====================
Unit tests for the digital_twin.network module (visualization and CLI).
"""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from digital_twin.network import (
    visualize_topology,
    TYPE_COLORS,
    TYPE_SHAPES,
    STATUS_BORDER,
    FIXED_POSITIONS,
)
from digital_twin.topology import create_campus_topology


class TestNetworkConstants:
    def test_type_colors_has_all_device_types(self):
        expected_types = {
            "External", "Security", "Monitoring", "Router",
            "Switch", "Server", "IoT", "Endpoint", "AccessPoint"
        }
        assert set(TYPE_COLORS.keys()) == expected_types

    def test_type_shapes_has_all_device_types(self):
        expected_types = {
            "External", "Security", "Monitoring", "Router",
            "Switch", "Server", "IoT", "Endpoint", "AccessPoint"
        }
        assert set(TYPE_SHAPES.keys()) == expected_types

    def test_status_border_has_all_statuses(self):
        expected_statuses = {"NORMAL", "WARNING", "SUSPICIOUS", "UNDER_ATTACK", "OFFLINE"}
        assert set(STATUS_BORDER.keys()) == expected_statuses

    def test_fixed_positions_has_all_devices(self):
        topo = create_campus_topology()
        device_ids = {d.device_id for d in topo.all_devices()}
        assert set(FIXED_POSITIONS.keys()) == device_ids


class TestVisualizeTopology:
    @patch("digital_twin.network.plt.savefig")
    @patch("digital_twin.network.plt.close")
    @patch("digital_twin.network.plt.subplots")
    def test_visualize_topology_creates_figure(self, mock_subplots, mock_close, mock_savefig):
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        topo = create_campus_topology()
        result = visualize_topology(topo, output_path="test_output.png", show=False)

        assert mock_subplots.called
        assert mock_savefig.called
        assert mock_close.called
        # Function returns absolute path
        assert result.endswith("test_output.png")

    @patch("digital_twin.network.plt.savefig")
    @patch("digital_twin.network.plt.close")
    @patch("digital_twin.network.plt.subplots")
    def test_visualize_topology_creates_output_dir(self, mock_subplots, mock_close, mock_savefig, tmp_path):
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        topo = create_campus_topology()
        output_path = tmp_path / "subdir" / "topology.png"

        result = visualize_topology(topo, output_path=str(output_path), show=False)
        assert Path(result).name == "topology.png"


class TestNetworkCLI:
    @patch("digital_twin.network.visualize_topology")
    @patch("digital_twin.network.create_campus_topology")
    @patch("builtins.print")
    def test_main_runs_without_error(self, mock_print, mock_create_topo, mock_visualize):
        mock_topo = MagicMock()
        mock_topo.summary.return_value = {
            "total_devices": 26,
            "total_connections": 25,
            "status_counts": {"NORMAL": 26}
        }
        mock_topo.all_devices.return_value = []
        mock_create_topo.return_value = mock_topo
        mock_visualize.return_value = "/tmp/test.png"

        from digital_twin.network import main
        main()

        assert mock_create_topo.called
        assert mock_visualize.called