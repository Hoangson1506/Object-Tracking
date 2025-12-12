import cv2
import numpy as np

class LicensePlateRecognizer:
    """
    Recognize license plate of violated vehicles
    """
    def __init__(self, license_model, character_model):
        self.license_model = license_model
        self.character_model = character_model

    def update(self, vehicle, frame):
        """
        Detect + OCR license plate for a single vehicle
        Returns candidate license plate string (NOT final)
        """
        print('update_license')
        if frame is None:
            return None

        x1, y1, x2, y2 = map(int, vehicle.get_state()[0])
        h, w, _ = frame.shape

        crop = frame[
            max(0, y1):min(h, y2),
            max(0, x1):min(w, x2)
        ].copy()

        if crop.size == 0:
            return None

        results = self.license_model.predict(crop, verbose=False)
        if len(results) == 0 or len(results[0].boxes) == 0:
            print('0 results')
            return None

        box = max(results[0].boxes, key=lambda b: b.conf)[0]
        lx1, ly1, lx2, ly2 = map(int, box.xyxy[0].cpu().numpy())
        lp_crop = crop[ly1:ly2, lx1:lx2]

        if lp_crop.size == 0:
            return None

        plate_text = self._ocr(lp_crop)

        if plate_text is None or len(plate_text) <= 3:
            return None

        return plate_text.strip()


    def _ocr(self, lp_img):
        """
        OCR using your character model.
        Modify this based on the OCR you're using.
        """
        try:
            result = self.character_model.ocr(lp_img)
            if isinstance(result, str):
                return result

            if isinstance(result, list) and len(result) > 0:
                return result[0][1][0]  # text

        except Exception as e:
            print("[OCR Error]", e)
            return None

        return None