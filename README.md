# Face Match Demo

A local Streamlit interface backed by FastAPI and DeepFace. Capture two images to compare them. If DeepFace reports a match, the UI also displays the age, gender, race, and emotion analysis returned for each image, along with the full verification metadata.

Install Python 3.11 before setup. The workspace's Python 3.14 environment does not have DeepFace installed and may not be supported by the TensorFlow build required for inference.

## Run on Windows

Create and activate an environment, then install the dependencies:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Start the API in one terminal:

```powershell
uvicorn api:app --reload
```

Start Streamlit in a second terminal, from the project directory and with the same environment activated:

```powershell
streamlit run app.py
```

Open the local URL printed by Streamlit. The first verification may take longer because DeepFace downloads and initializes its model weights. The API listens on `http://127.0.0.1:8000`; set `FACE_API_URL` to change the URL used by Streamlit.

Phone cameras are supported by the two camera controls in the app. Open the app from the phone using an HTTPS URL and allow camera access; mobile browsers generally block live camera access on a LAN `http://` URL. A secure HTTPS deployment or reverse proxy is needed for phone access outside the local computer.

Only use images you have permission to process. Uploaded images are processed in memory and are not written to disk by this app.
