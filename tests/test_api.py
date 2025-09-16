import requests
import pytest

# Direct container endpoints
ENDPOINTS = [
    "http://localhost:8000",   # api
    "http://localhost:8001",   # api-1
    "https://dataml.xyz"       # Nginx load balancer / domain
]

@pytest.mark.parametrize("base_url", ENDPOINTS)
def test_health_check(base_url):
    """Check if /health endpoint works on all services"""
    response = requests.get(f"{base_url}/health", timeout=5)
    assert response.status_code == 200
    assert "status" in response.json()
    assert response.json()["status"] == "ok"

@pytest.mark.parametrize("base_url", ENDPOINTS)
def test_root_endpoint(base_url):
    """Check if root endpoint returns valid response"""
    response = requests.get(f"{base_url}/", timeout=5)
    # Depending on your app this may be 200 or 404
    assert response.status_code in [200, 404]
