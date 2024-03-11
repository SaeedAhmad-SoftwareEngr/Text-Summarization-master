import http.client
import json

def generate_summary_rapidapi(input_text, rapidapi_key):
    conn = http.client.HTTPSConnection("open-ai21.p.rapidapi.com")

    payload = {
        "text": input_text
    }

    headers = {
        'content-type': "application/json",
        'X-RapidAPI-Key': "11eb9681d3msh6455a6f3e4b9bd8p159b27jsn14d646b2c892",
        'X-RapidAPI-Host': "open-ai21.p.rapidapi.com"
    }

    conn.request("POST", "/summary", json.dumps(payload), headers)
    res = conn.getresponse()
    data = res.read()

    return data.decode("utf-8")
