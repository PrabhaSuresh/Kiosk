import requests

api_url = "" 
response_url = "" 

payload = {
    "input_text": "What are the symptoms of diabetes?",
    "response_url": response_url
}

response = requests.post(api_url, json=payload)
print(response.json())
