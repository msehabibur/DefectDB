import streamlit as st
from google.oauth2 import service_account
from googleapiclient.discovery import build

st.title("Google Drive Access Test")

try:
    creds = service_account.Credentials.from_service_account_info(
        dict(st.secrets["gdrive_service_account"]),
        scopes=["https://www.googleapis.com/auth/drive.readonly"],
    )
    st.success("✅ Service account loaded successfully.")
except Exception as e:
    st.error(f"❌ Failed to load service account: {e}")
    st.stop()

try:
    service = build("drive", "v3", credentials=creds, cache_discovery=False)
    folder_id = "1gYTtFpPIRCDWpLBW855RA6XwG0buifbi"
    results = service.files().list(
        q=f"'{folder_id}' in parents and trashed=false",
        fields="files(id, name, mimeType)"
    ).execute()
    files = results.get("files", [])
    if not files:
        st.warning("No files found or access denied. Make sure folder is shared with the service account.")
    else:
        st.write("**Files in your folder:**")
        for f in files:
            st.write(f"📄 {f['name']} — {f['id']}")
except Exception as e:
    st.error(f"❌ Failed to connect to Google Drive: {e}")
