import requests

filename = "Chapter10.MontesinosLpez2022_Chapter_FundamentalsOfArtificialNeural.pdf"
url = f"http://localhost:8000/documents/delete?filename={filename}"

try:
    response = requests.delete(url)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text}")
except Exception as e:
    print(f"Error: {e}")
