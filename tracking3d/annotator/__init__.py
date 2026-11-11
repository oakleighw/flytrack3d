"""The annotator of the guided 3D tracks (main_tracks3d_guided.py).

    annotation_tool  start / end point annotation across the views of a session
    suggest          points suggested in a view from those of earlier views (epipolar geometry)
"""

from .annotation_tool import ViewInput, run_annotation_gui, view_input_from_source, view_inputs_from_session

__all__ = ["ViewInput", "run_annotation_gui", "view_input_from_source", "view_inputs_from_session"]
