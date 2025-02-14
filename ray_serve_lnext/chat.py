import requests


print(requests.post("http://localhost:8000/", json={"prompt": "What is in this image?", "url": "https://t4.ftcdn.net/jpg/07/08/47/75/360_F_708477508_DNkzRIsNFgibgCJ6KoTgJjjRZNJD4mb4.jpg"}).text)

