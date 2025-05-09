Installation
============

Prerequisites
-------------

* **Python ≥ 3.8**
* **OpenCV ≥ 4.8** compiled with *DNN* and *ONNX* support
* **NumPy ≥ 1.23**

.. note::

   If you install the pre‑built ``opencv‑python`` wheels from PyPI, ONNX support is already enabled.


Install from source
-------------------

Clone the repository and install in editable / developer mode:

.. code-block:: bash

   git clone https://github.com/brahimself/facebox.git
   cd facebox
   pip install -e


Model assets
------------

Three binary assets are required at runtime:

==================================  ==========================================================
Filename                             Purpose
==================================  ==========================================================
``haarcascade_frontalface_default.xml``   Haar cascade for frontal face detection
``openface.nn4.small2.v1.t7``             Face‑embedding network (128‑D)
``emotion-ferplus-8.onnx``                Emotion classification network (FER+)
==================================  ==========================================================
