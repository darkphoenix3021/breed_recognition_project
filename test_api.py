import requests

# The URL where your Flask API is running
API_URL = 'http://127.0.0.1:5000/predict'

# The path to the image you want to test
# Make sure this image exists!
IMAGE_PATH = 'dataset/Gir/gir_image_1.jpg'

try:
    # Open the image file in binary mode
    with open(IMAGE_PATH, 'rb') as image_file:
        files = {'file': (IMAGE_PATH, image_file, 'image/jpeg')}
        
        # Send the POST request
        response = requests.post(API_URL, files=files)
        
        # Print the results
        print(f"Status Code: {response.status_code}")
        print("Response JSON:")
        print(response.json())

except FileNotFoundError:
    print(f"Error: The file '{IMAGE_PATH}' was not found.")
except requests.exceptions.ConnectionError as e:
    print(f"Error: Could not connect to the API server at {API_URL}.")
    print("Please ensure the 'run_api.py' server is running in a separate terminal.")
