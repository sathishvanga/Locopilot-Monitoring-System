# Task 0014: Hardcoded S3 upload API URL

- **Issue ID:** C-07
- **Priority:** Phase 3 - Security & Reliability (Item 14)
- **Severity:** CRITICAL
- **Category:** Configuration / Security
- **File:** `app/services/s3_upload_service.py:28`

## Description

`self.api_url = "https://api.mindcoinapps.com/ai_demo_api/amazonUpload/uploadWithFolder"` is hardcoded. Cannot be changed per environment without code modification.

## Fix

Move to `config.py` settings as `s3_upload_api_url` and reference via `self.settings.s3_upload_api_url`.
