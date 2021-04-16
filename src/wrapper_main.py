# Add ./hent-AI into the path, so that imports work.
from sys import path
from os import getcwd
path.append(getcwd() + "/hent-AI")

# Local libraries.
from wrapper_detector import Detector

# External libraries.
from PIL import Image
import redis
import io

def main():
    r = redis.Redis(host="localhost", port=6379, db=0)

    detector_and_decensor_instance = Detector("/models/hent-AI model 268/weights.h5")
    detector_and_decensor_instance.load_weights()

    # Test the connection to Redis.
    r.get("connection-test")

    while True:
        print("ready to receive censored image")
        key, uuid = r.blpop(["censored-images:hent-ai:bar", "censored-images:hent-ai:mosaic"], 0)
        print("received censored image")

        censored_img_data = r.get("censored-images:%s" % uuid.decode())

        censored_img_file = io.BytesIO(censored_img_data)
        censored_img_file.seek(0)

        censored_img = Image.open(censored_img_file)

        if key.decode() == "censored-images:hent-ai:bar":
            prepared_img = detector_and_decensor_instance.detect_and_cover(censored_img, 3)

            prepared_img_file = io.BytesIO()
            prepared_img.save(prepared_img_file, format="PNG")

            r.set("censored-images:%s" % uuid.decode(), prepared_img_file.getvalue())
            r.rpush("censored-images:deepcreampy:bar", "%s" % uuid.decode())
        else:
            print("can't do mosaic yet")

if __name__ == "__main__":
    main()
