Installation
============

Prerequisites
-------------

* **Python ≥ 3.8**
* **OpenCV ≥ 4.8** compiled with *DNN* and *ONNX* support
* **NumPy ≥ 1.23**

.. note::

   If you install the pre‑built ``opencv‑python`` wheels from PyPI, ONNX support is already enabled.

Install from PyPI (recommended)
-------------------------------

.. code-block:: bash

   pip install facebox               # (Coming soon: publish to PyPI!)

Install from source
-------------------

Clone the repository and install in editable / developer mode:

.. code-block:: bash

   git clone https://github.com/your‑org/facebox.git
   cd facebox
   pip install -e ".[dev]"

The extra ``[dev]`` group pulls in lint/test tools such as **black**, **ruff** and **pytest**.

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

If these files do not ship with your package manager, download them manually and place them in the project root
(or anywhere on ``$PYTHONPATH`` that your code can locate).
