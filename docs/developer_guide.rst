Developer Guide
===============

Project layout
--------------

::

   facebox/
   ├─ facebox/                 # library source
   │  ├─ __init__.py
   │  ├─ detector.py           # face detection + embeddings
   │  └─ emotion.py            # FER+ emotion classifier
   ├─ run.py                   # webcam demo
   ├─ docs/                    # Sphinx documentation
   ├─ notebooks/               # exploratory Jupyter notebooks
   └─ setup.py

Algorithmic pipeline
--------------------

.. mermaid::

   graph LR
       A[Video Frame] --> B[Haar Cascade<br/>Detection]
       B --> C[Crop Face ROI]
       C --> D[OpenFace<br/>Embedding]
       C --> E[FER+ ONNX<br/>Emotion]
       D --> F{Compare<br/>Embeddings}
       F -->|< 0.85| G[Identity]
       F -->|≥ 0.85| H[Unknown]
       G --> I[Overlay<br/>Text]
       E --> I
       H --> I
       I --> J[Display]

Threshold tuning
~~~~~~~~~~~~~~~~

The *0.85* L2‑distance threshold trades off **precision** vs **recall**.  
Adjust according to your dataset size and lighting conditions.

Testing
-------

Run the unit test suite:

.. code-block:: bash

   pytest -q

Tests live beside modules in ``tests/`` and exercise:

* face detection speed / accuracy (synthetic image fixtures)
* embedding distance distribution on a mini‑dataset
* emotion inference outputs

Continuous Integration (CI) hooks run **ruff**, **black**, and **pytest** on every PR.
