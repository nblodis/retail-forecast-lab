# Data and publishing safety

- Only synthetic examples belong in the public repository.
- Do not commit production extracts, SQL connection strings, company schemas, customer identifiers, internal hostnames, secrets, local user paths or unpublished metrics.
- This package deliberately uses a generic local CSV interface and never connects to corporate systems.
- `outputs/`, `.env*`, SQLite files and Excel exports are gitignored, but always inspect `git diff --cached` before committing.
- If you accidentally publish sensitive material, remove public access, revoke affected credentials, and follow the relevant organization's incident response process; deleting a commit is not sufficient to erase its history.