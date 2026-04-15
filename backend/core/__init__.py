"""backend.core — UI-free canonical services and types shared by TUI and desktop GUI.

Rules:
  * Nothing in this package imports rich, textual, PySide6, or any UI framework.
  * Service signatures change by addition only while the GUI is under construction.
  * Existing TUI/classic call sites are migrated opportunistically, never in a
    standalone refactor PR.
"""
