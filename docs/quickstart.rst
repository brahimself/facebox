Quick‑Start
===========

Fire up the *demo* to see FaceBox in action:

.. code-block:: bash

   python run.py

A webcam window appears.  
* Press **q** to quit.  
* Move closer / farther to observe bounding‑box scaling.  
* Smile, frown or grimace to watch the emotion overlay change.

Code walk‑through
-----------------

1. A ``cv2.VideoCapture`` stream pulls frames from your default webcam.
2. ``CascadeClassifier`` locates faces in grayscale.
3. Each face region is:

   * forwarded through **OpenFace** → *128‑D embedding*
   * compared (L2 norm) to the stored *signature* vectors
   * forwarded through **FER+** → *emotion logits*

4. Results are painted on‑screen: *bounding box → identity → emotion → per‑class probability bar*.
