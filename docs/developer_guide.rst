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
   ├─ notebooks/               # exploratory colab notebooks
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
