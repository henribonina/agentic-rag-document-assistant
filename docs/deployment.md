# Deployment and verification

## Streamlit Community Cloud

1. Confirm the complete source is committed to a GitHub repository.
2. Open Streamlit Community Cloud and sign in with GitHub.
3. Select **Create app** and choose:
   - Repository: the project repository
   - Branch: `main`
   - Main file path: `app.py`
4. Open **Advanced settings** and add the following secret:

   ```toml
   OPENAI_API_KEY = "your_api_key_here"
   ```

5. Select **Deploy** and wait for dependency installation and application startup.
6. Keep the secret out of GitHub, screenshots, reports, and submission ZIP files.

## Deployment verification

Complete this checklist after deployment:

- [ ] The application opens without an error.
- [ ] A readable TXT file can be uploaded and processed.
- [ ] The interface reports searchable chunks and indexed vectors.
- [ ] A focused question retrieves at least one supporting passage.
- [ ] The answer includes a valid inline source label such as `[S1]`.
- [ ] Supporting evidence displays the correct source file and passage.
- [ ] The agent workflow shows planning, retrieval, reasoning, and validation.
- [ ] The safety tab completes its offline regression checks.

## Local verification

On Windows:

```powershell
py -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
copy .env.example .env
python -m streamlit run app.py
```

Add the API key only to the local `.env` file. Open `http://localhost:8501`,
upload a readable document, and ask a question based on its contents. Press
`Ctrl+C` in the terminal to stop the local server.

## Limitations

- Image-only or scanned PDFs require OCR before upload.
- The vector store is in memory and does not persist across restarts.
- Supported formats are PDF, TXT, CSV, and XLSX.
- Important answers require human review against the displayed evidence.
- API availability, model access, rate limits, and account credits affect online
  embedding and generation.

## Troubleshooting

- **No selectable PDF text:** run OCR or upload a text-based copy.
- **Missing API key:** add `OPENAI_API_KEY` to local `.env` or deployment secrets.
- **Insufficient quota:** confirm API billing and available credits.
- **No relevant passages:** ask a more focused question or adjust the retrieval
  count and chunk settings.
- **Dependency failure:** redeploy after confirming `requirements.txt` is present
  at the repository root.

## Submission package

Include source code, `requirements.txt`, `.env.example`, `README.md`, and the
`docs` and `tests` directories. Exclude `.env`, `.venv`, API keys, uploaded
private files, `__pycache__`, `.pytest_cache`, and vector database files.
