# Setting up a workshop on Renku

This guide is for organisers. Participants get a public Renku project, built from this repository, where the notebooks run with a shared OpenAI key. The key is never stored in the repository: it sits in a password-protected PolyBox folder that Renku mounts into each session.

Throughout this guide, `acronym` is a short identifier for the workshop or partner (uppercase where shown as `ACRONYM`).

## 1. OpenAI API key

1. Go to the OpenAI Platform and select the project `ei26-02_educationgeneral_gpt-4o-mini`. It allows `gpt-4o-mini` and `text-embedding-3-small`, which covers everything the notebooks use. If you don't have access, ask support@datascience.ch.
2. Under "API Keys", create a new secret key:
   - Name: `ACRONYM_DDMMYYYY` (date of the event).
   - Expiration: the evening of the event or the day after.
   - Permissions: All.
3. Copy the key; it is shown only once. Set a reminder to revoke it right after the workshop.

## 2. PolyBox folder with the key

1. On [PolyBox](https://polybox.ethz.ch), create a folder `ki-kompass-acronym`.
2. Inside it, create a plain text file named `openai_api_key` (PolyBox's "New text file" adds `.txt`; `openai_api_key.txt` works too). The file contains only the key on one line, for example `sk-proj-abc123...`: no quotes, no `OPENAI_API_KEY=` prefix. A trailing newline is fine.
3. Share the folder with a public link: "Share link", then set a password (for example `acronymYYYY`) and an expiration date after the workshop. Copy the link.

Use the restricted OpenAI project, a short expiration and a budget limit, create a new key for each workshop, and revoke it right after the event.

## 3. Renku project

1. In the Renku group (for example `agentic-workshop`, or your own namespace), create a project:
   - Name: `ki-kompass-acronym`.
   - Visibility: Public.
2. Connect the repository `https://github.com/SwissDataScienceCenter/ki-kompass-conversational-toolkit`. It is public, so no token is needed.
3. Under "Data connectors", create a new data connector:
   - Provider: PolyBox, in shared mode. Enter the public link and password, click "Test connection", then continue. (If PolyBox is not listed, use WebDAV with the same link and password.)
   - Name: `tokens-acronym-polybox`, visibility Public.
   - In "Advanced settings", mount it on `tokens`.
4. Create a session launcher: "Session launcher", then "Create from code". The image is built from `pyproject.toml` and `uv.lock` in this repository; the first build takes a while.
5. Choose the session size. The workshop notebooks need about 2 CPU, 8 GB memory and 20 GB storage, no GPU. The optional image part of T03 needs more memory and is not covered in the workshop.

At session start, `init.d/setup_project.sh` finds `openai_api_key` in the mounted folder and sets `OPENAI_API_KEY` for terminals and notebook kernels. If the key is missing, check `~/setup_project.log` in the session.

## 4. Test

Open the project in a private browser window (not logged in) and launch a session. Check that:

- the repository is in the session and `backend/notebooks/english/` and `backend/notebooks/german/` are there,
- `T01_baseline_rag.ipynb` runs top to bottom.

## 5. Booking

Announce the event with Renku through the Notion form ("Announce your course or event with Renku"). Use the Renku URL of the project and the session size you chose in step 3.
