"""
Tests for the metrics calculation module.
"""

import numpy as np
import pytest
from unittest.mock import Mock
from src.train.metrics import (
    count_nearby_apples,
    check_ate_last_apple_in_cluster,
    compute_agent_step_metrics,
)
from src.env.commons_env import APPLE_RADIUS


class TestCountNearbyApples:
    """Tests for count_nearby_apples function."""
    
    def test_no_apples(self):
        """Test with no apples in view."""
        view = np.array([[' ', ' ', ' '],
                         [' ', '1', ' '],  # Agent is represented as '1'
                         [' ', ' ', ' ']])
        count = count_nearby_apples(view, view_center_row=1, view_center_col=1, radius=2)
        assert count == 0
    
    def test_apple_at_center(self):
        """Test with apple at agent position (agent would see itself, not the apple)."""
        # Note: In reality, if agent is at a position with an apple, the apple is consumed
        # and the agent sees itself. This test checks the counting logic.
        view = np.array([[' ', ' ', ' '],
                         [' ', 'A', ' '],
                         [' ', ' ', ' ']])
        count = count_nearby_apples(view, view_center_row=1, view_center_col=1, radius=2)
        assert count == 1
    
    def test_apple_within_radius(self):
        """Test with apple within radius."""
        view = np.array([[' ', ' ', 'A', ' '],
                         [' ', '1', ' ', ' '],  # Agent at center
                         [' ', ' ', ' ', ' ']])
        count = count_nearby_apples(view, view_center_row=1, view_center_col=1, radius=2)
        # Distance from (1,1) to (0,2) = sqrt(1^2 + 1^2) = sqrt(2) ≈ 1.41 <= 2
        assert count == 1
    
    def test_apple_outside_radius(self):
        """Test with apple outside radius."""
        view = np.array([['A', ' ', ' ', ' ', ' ', ' '],
                         [' ', ' ', ' ', ' ', ' ', ' '],
                         [' ', ' ', '1', ' ', ' ', ' '],  # Agent at center
                         [' ', ' ', ' ', ' ', ' ', ' ']])
        count = count_nearby_apples(view, view_center_row=2, view_center_col=2, radius=2)
        # Distance from (2,2) to (0,0) = sqrt(2^2 + 2^2) = sqrt(8) ≈ 2.83 > 2
        assert count == 0
    
    def test_multiple_apples(self):
        """Test with multiple apples within radius."""
        view = np.array([[' ', 'A', ' '],
                         ['A', '1', 'A'],  # Agent at center, apples around
                         [' ', 'A', ' ']])
        count = count_nearby_apples(view, view_center_row=1, view_center_col=1, radius=2)
        # All 4 apples are within radius 2
        assert count == 4
    
    def test_mixed_apples(self):
        """Test with some apples within and some outside radius."""
        view = np.array([['A', ' ', ' ', ' ', 'A'],
                         [' ', ' ', '1', ' ', ' '],  # Agent at center
                         [' ', 'A', ' ', ' ', ' ']])
        count = count_nearby_apples(view, view_center_row=1, view_center_col=2, radius=2)
        # (0,0): dist = sqrt(1^2 + 2^2) = sqrt(5) ≈ 2.24 > 2 (outside)
        # (0,4): dist = sqrt(1^2 + 2^2) = sqrt(5) ≈ 2.24 > 2 (outside)
        # (2,1): dist = sqrt(1^2 + 1^2) = sqrt(2) ≈ 1.41 <= 2 (inside)
        assert count == 1


class TestCheckAteLastAppleInCluster:
    """Tests for check_ate_last_apple_in_cluster function."""
    
    def test_no_spawn_points(self):
        """Test with no spawn points."""
        agent_pos = np.array([5, 5])
        apple_points = []
        world_map = np.array([[' '] * 10] * 10)
        result = check_ate_last_apple_in_cluster(agent_pos, apple_points, world_map)
        assert result is False
    
    def test_spawn_point_far_away(self):
        """Test when spawn point is far from agent position."""
        agent_pos = np.array([5, 5])
        apple_points = [[0, 0]]  # Far away
        world_map = np.array([[' '] * 10] * 10)
        result = check_ate_last_apple_in_cluster(agent_pos, apple_points, world_map)
        assert result is False
    
    def test_last_apple_in_cluster(self):
        """Test when agent ate the last apple in a cluster."""
        agent_pos = np.array([5, 5])
        apple_points = [[5, 5]]  # Agent is at spawn point
        world_map = np.array([[' '] * 10] * 10)  # No apples remaining
        # Place one apple at spawn point, then it gets eaten
        world_map[5, 5] = ' '  # Already eaten
        
        result = check_ate_last_apple_in_cluster(agent_pos, apple_points, world_map)
        assert result is True
    
    def test_not_last_apple_in_cluster(self):
        """Test when other apples remain in cluster."""
        agent_pos = np.array([5, 5])
        apple_points = [[5, 5]]
        world_map = np.array([[' '] * 10] * 10)
        world_map[5, 5] = ' '  # Eaten apple
        world_map[5, 6] = 'A'  # Another apple nearby (within APPLE_RADIUS=2)
        
        result = check_ate_last_apple_in_cluster(agent_pos, apple_points, world_map)
        assert result is False
    
    def test_multiple_spawn_points(self):
        """Test with multiple spawn points, finding nearest."""
        agent_pos = np.array([5, 5])
        apple_points = [[0, 0], [5, 5], [10, 10]]  # Middle one is nearest
        world_map = np.array([[' '] * 11] * 11)
        world_map[5, 5] = ' '  # Eaten
        
        result = check_ate_last_apple_in_cluster(agent_pos, apple_points, world_map)
        assert result is True
    
    def test_cluster_with_apples_at_spawn_radius(self):
        """Test cluster detection with apples at spawn radius boundary."""
        agent_pos = np.array([5, 5])
        apple_points = [[5, 5]]
        world_map = np.array([[' '] * 10] * 10)
        world_map[5, 5] = ' '  # Eaten
        # Place apple at distance exactly APPLE_RADIUS (should be included)
        world_map[5, 5 + APPLE_RADIUS] = 'A'  # At radius boundary
        
        result = check_ate_last_apple_in_cluster(agent_pos, apple_points, world_map)
        # Should be False because there's still an apple in the cluster
        assert result is False


class TestComputeAgentStepMetrics:
    """Tests for compute_agent_step_metrics function."""
    
    def test_no_apple_eaten(self):
        """Test metrics when no apple is eaten."""
        agent = Mock()
        agent.get_state.return_value = np.array([[' ', ' ', ' '],
                                                 [' ', '1', ' '],  # Agent represented as '1'
                                                 [' ', ' ', ' ']])
        agent.row_size = 1
        agent.col_size = 1
        agent.get_pos.return_value = np.array([5, 5])
        
        env = Mock()
        env.apple_points = [[5, 5]]
        env.world_map = np.array([[' '] * 10] * 10)
        
        metrics = compute_agent_step_metrics(
            agent=agent,
            env=env,
            reward=0.0,
            apple_eaten=False,
            nearby_radius=2
        )
        
        assert metrics["nearby_apples"] == 0
        assert metrics["ate_last_apple_in_cluster"] is False
    
    def test_apple_eaten_with_nearby_apples(self):
        """Test metrics when apple is eaten and nearby apples exist."""
        agent = Mock()
        agent.get_state.return_value = np.array([[' ', 'A', ' '],
                                                 ['A', '1', 'A'],  # Agent at center, apples around
                                                 [' ', 'A', ' ']])
        agent.row_size = 1
        agent.col_size = 1
        agent.get_pos.return_value = np.array([5, 5])
        
        env = Mock()
        env.apple_points = [[5, 5]]
        world_map = np.array([[' '] * 10] * 10)
        world_map[5, 5] = ' '  # Eaten
        world_map[5, 6] = 'A'  # Another apple nearby
        env.world_map = world_map
        
        metrics = compute_agent_step_metrics(
            agent=agent,
            env=env,
            reward=1.0,
            apple_eaten=True,
            nearby_radius=2
        )
        
        assert metrics["nearby_apples"] == 4  # 4 apples in view
        assert metrics["ate_last_apple_in_cluster"] is False  # Other apple remains
    
    def test_ate_last_apple_in_cluster(self):
        """Test when agent ate the last apple in cluster."""
        agent = Mock()
        agent.get_state.return_value = np.array([[' ', ' ', ' '],
                                                 [' ', '1', ' '],  # Agent at center
                                                 [' ', ' ', ' ']])
        agent.row_size = 1
        agent.col_size = 1
        agent.get_pos.return_value = np.array([5, 5])
        
        env = Mock()
        env.apple_points = [[5, 5]]
        world_map = np.array([[' '] * 10] * 10)
        world_map[5, 5] = ' '  # Eaten, no other apples
        env.world_map = world_map
        
        metrics = compute_agent_step_metrics(
            agent=agent,
            env=env,
            reward=1.0,
            apple_eaten=True,
            nearby_radius=2
        )
        
        assert metrics["nearby_apples"] == 0
        assert metrics["ate_last_apple_in_cluster"] is True

