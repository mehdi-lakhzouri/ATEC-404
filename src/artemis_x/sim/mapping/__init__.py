"""R4 local perception and mapping for the Webots controller.

Standard library only (the controller runs under Webots' system Python without numpy). Every live computation uses the
rover's ESTIMATED pose (wheel odometry); the Webots supervisor ground truth is never read here - evaluation tools that
need it live outside this package (``scripts/run_webots_r4.py``). ``tests/test_mapping_frames.py`` enforces this.
"""
