# Run the Streamlit UI from the project root (no need to activate the venv).
Set-Location $PSScriptRoot
& "$PSScriptRoot\venv\Scripts\python.exe" -m streamlit run app/app.py
