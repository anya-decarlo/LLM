import requests

DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"  # Replace with actual endpoint
API_KEY = "sk-614f19b5323e4778bb8e30474489b561"  # Replace with actual API key

headers = {"Authorization": f"Bearer {API_KEY}"}
response = requests.get(DEEPSEEK_API_URL, headers=headers)

if response.status_code == 200:
    print("✅ API Key is valid!")
    print("Response:", response.json())  # This might show account balance
else:
    print(f"❌ API error: {response.status_code} - {response.text}")