"""Webots simulation support for the Artemis-X rover (v0.1: Koala platform, scripted motion, RGB + depth capture).

Pure Python (no numpy) so it runs under the interpreter Webots launches for controllers. The Webots ``controller``
module is only imported by ``rover.run_controller``; ``drive`` and ``sensors`` are testable without Webots.
"""
