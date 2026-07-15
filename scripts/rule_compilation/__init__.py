"""Offline pipeline that compiles source manuals into the committed rule store.

These scripts run outside the runtime package. They may import ``cipoc.llm`` and
``cipoc.models``; the runtime never imports them. See ``compile_manual.py`` for
the driver and the plan's §5 for the pipeline design.
"""
