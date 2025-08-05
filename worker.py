from app import process_image
import time

if __name__ == "__main__":
    print("Worker started.")
    while True:
        # Ide jöhetne például a Cloudinary image URL-ek monitorozása vagy queue polling.
        time.sleep(10)
