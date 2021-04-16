# Add ./hent-AI into the path, so that imports work.
from sys import path
from os import getcwd
path.append(getcwd() + "/hent-AI")

# Local libraries.
from wrapper_detector import Detector

# External libraries.
import redis
import io
from erogaki_wrapper_shared_python.ImageProcessor import ImageProcessor
from NoCensoredRegionsFoundError import NoCensoredRegionsFoundError

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

        if key.decode() == "censored-images:hent-ai:bar":
            try:
                prepared_img = detector_and_decensor_instance.detect_and_cover(ImageProcessor.bytes_to_image(censored_img_data), 3)

                r.set("censored-images:%s" % uuid.decode(), ImageProcessor.image_to_bytes(prepared_img))
                r.rpush("censored-images:deepcreampy:bar", "%s" % uuid.decode())
            except NoCensoredRegionsFoundError as e:
                print(e.description)
                r.set("errors:%s" % uuid.decode(), e.json)
        else:
            print("can't do mosaic yet")

if __name__ == "__main__":
    main()
