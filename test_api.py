import json
from app import app

with app.test_client() as client:
    resp = client.get('/api/run')
    print('API Status:', resp.status_code)
    data = resp.get_json()
    if data:
        print('Keys:', sorted(data.keys()))
    else:
        print('No JSON returned')

