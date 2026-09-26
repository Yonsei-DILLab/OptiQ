"""Shared trajectory visibility defaults requested on 2026-09-26."""

TRAJECTORY_LINEWIDTH = 1.8
TRAJECTORY_ALPHA = .5
TRAJECTORY_COLOR = "#c414c9"
LEARNING_LINEWIDTH = 2.4

# Outcome colors use saved terminal goal IDs, not visual proximity to a goal.
SUCCESS_TRAJECTORY_COLOR = "#65d6bd"
FAILURE_TRAJECTORY_COLOR = "#ed80b0"
VISITED_GOAL_COLOR = "#17429c"
UNVISITED_GOAL_COLOR = "#a51d2d"
UNKNOWN_GOAL_COLOR = "#cccccc"
GOAL_BORDER_COLOR = "#ffffff"
GOAL_BORDER_LINEWIDTH = 1.1

TRAJECTORY_PALETTES = {
    "mint-pink": (SUCCESS_TRAJECTORY_COLOR, FAILURE_TRAJECTORY_COLOR),
    "blue-red": ("#2065d1", "#e74747"),
    "sky-yellow": ("#58b9e6", "#efc638"),
    "sky-orange": ("#58b9e6", "#f28c38"),
}
