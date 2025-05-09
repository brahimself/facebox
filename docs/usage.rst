Usage Guide
===========

Import the library
------------------

.. code-block:: python

   import cv2
   from facebox.detector import load_signatures, get_signature
   from facebox.emotion import detect_emotion

Signature enrolment
-------------------

.. code-block:: python

   elon_imgs  = ["elon1.jpg", "elon2.jpg", "elon3.jpg"]
   larry_imgs = ["larry1.png", "larry2.png", "larry3.png"]

   signatures = {{
       "Elon":  load_signatures(elon_imgs,  "Elon"),
       "Larry": load_signatures(larry_imgs, "Larry"),
   }}

Real‑time frame processing
--------------------------

.. code-block:: python

   # read webcam frame
   ok, frame = cap.read()
   gray      = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   faces     = face_cascade.detectMultiScale(gray, 1.3, 5)

   for (x, y, w, h) in faces:
       sub = frame[y:y+h, x:x+w]
       emb = get_signature(sub)
       name = "Unknown"
       for label, ref in signatures.items():
           if np.linalg.norm(ref - emb) < 0.85:
               name = label
               break

       emotion = detect_emotion(gray, (x, y, w, h))
       # draw graphics …
