# test_upload_endpoints.py
import os, requests, uuid
SUPA_URL = "https://cdhvpmulmvgwaqmjcmtl.supabase.co"   # your project
BUCKET = "uploads"
SERVICE_KEY = os.environ.get("SUPA_SERVICE_KEY") or "YOUR_SERVICE_ROLE_KEY_HERE"
FILE_PATH = r"D:\Downloads\ncr_ride_bookings_reduced.csv"  # update if needed

assert SERVICE_KEY and SERVICE_KEY != "YOUR_SERVICE_ROLE_KEY_HERE", "Set SUPA_SERVICE_KEY env var or edit the script."

with open(FILE_PATH, "rb") as fh:
    file_bytes = fh.read()

headers = {
    "Authorization": f"Bearer {SERVICE_KEY}",
    "apikey": SERVICE_KEY,
}

unique_name = f"{uuid.uuid4().hex}_{os.path.basename(FILE_PATH)}"
files = {"file": (os.path.basename(FILE_PATH), file_bytes)}

# Candidate URL A: ?name= style
url_a = f"{SUPA_URL}/storage/v1/object/{BUCKET}"
params_a = {"name": unique_name}
print("Trying URL A:", url_a, "params:", params_a)
resp_a = requests.post(url_a, headers=headers, params=params_a, files=files, timeout=60)
print("A status:", resp_a.status_code)
print("A request URL:", resp_a.request.url)
print("A body:", resp_a.text)
print("-" * 80)

# Candidate URL B: /upload style
url_b = f"{SUPA_URL}/storage/v1/object/{BUCKET}/upload"
params_b = {"name": unique_name}
print("Trying URL B:", url_b, "params:", params_b)
resp_b = requests.post(url_b, headers=headers, params=params_b, files=files, timeout=60)
print("B status:", resp_b.status_code)
print("B request URL:", resp_b.request.url)
print("B body:", resp_b.text)
print("-" * 80)

# Candidate URL C: older SDKs sometimes use /object/uploads (plural) - try it too
url_c = f"{SUPA_URL}/storage/v1/object/{BUCKET}/uploads"
print("Trying URL C:", url_c, "params:", params_a)
resp_c = requests.post(url_c, headers=headers, params=params_a, files=files, timeout=60)
print("C status:", resp_c.status_code)
print("C request URL:", resp_c.request.url)
print("C body:", resp_c.text)
print("-" * 80)

headers = {
    "Authorization": f"Bearer {SERVICE_KEY}",
    "apikey": os.environ.get("SUPABASE_ANON_KEY")  # use anon key here
}
