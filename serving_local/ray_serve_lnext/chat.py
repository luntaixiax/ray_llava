import requests


print(requests.post("http://localhost:8000/", json={"prompt": "What is in this image?", "url": "https://images.pexels.com/photos/45201/kitty-cat-kitten-pet-45201.jpeg"}).text)

