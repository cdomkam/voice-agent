def make_a_request(function_url: str, data: dict):
    from google.auth.transport.requests import Request
    from google.oauth2 import id_token
    import os
    import requests
    
    import dotenv
    CURRENT_DIR=os.path.dirname(os.path.abspath(__file__))

    dotenv.load_dotenv(dotenv_path="keys/keys.env")
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = CURRENT_DIR + os.environ["GEM_KEYS_FUNCTION"]
    
    def get_id_token(function_url: str):
        # Generate a custom token for a specific UID
        auth_req = Request()
        token = id_token.fetch_id_token(auth_req, function_url)
        return token
    
    token = get_id_token(function_url = function_url)
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-type': 'application/json'
    }
    
    response = requests.post(function_url, headers=headers, json=data)
    
    if response.status_code == 200:
        print('function call succeeded:', response.json())
        return response.json()
    else:
        print('function call failed:', response.text)