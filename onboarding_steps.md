# 🧪 Developer Onboarding – AaaS MVP Platform

Welcome to the Andrology-as-a-Service (AaaS) MVP GitHub repository. This document outlines how the development effort is structured into two clear phases, what issues to start with, and where to find the architecture overview.

---

## ✅ Phase 1 – MVP Core Modules (Target: July 14, 2025)

> Goal: Build the foundational system for sperm analysis with core reporting.

### Includes:
- 🧬 Count Estimation (image-based)
- 🧬 Motility Tracking (video-based)
- 🧬 Morphology Classification (SCIAN dataset)
- 🖥️ Frontend UI (upload image/video, view outputs)

### Start Here:
- [Issue #4 – Core Module Implementation](../../issues/4)
- [Issue #2 – Morphology Classification (SCIAN)](../../issues/2)

Focus is on local/static file-based implementation. No login system or trial gating in this phase.

---

## 🚀 Phase 2 – Extended Capabilities (Target: July 30, 2025)

> Goal: Add advanced features and cloud infrastructure.

### Includes:
- 🧠 Insight Engine (WHO-guided clinical messages)
- 🧪 DNA Fragmentation Module (SCD assay images)
- ☁️ Cloud Deployment (Render/Fly.io + Supabase)
- 🔐 Trial-based user login and expiration logic

### Relevant Issues:
- [Issue #5 – Insight Engine](../../issues/5)
- [Issue #6 – DFI Module](../../issues/6)
- [Issue #7 – Cloud Hosting + Login](../../issues/7)
- [Issue #8 – Architecture Diagram Upload](../../issues/8)

---

## 🗺️ Cloud Architecture

This MVP is designed for eventual cloud deployment with trial access for labs/andrologists.

![MVP Cloud Architecture](docs/architecture/cloud_mvp_architecture.png)

---

## 🧰 Tech Stack Summary

| Layer         | Stack                          |
|---------------|--------------------------------|
| Backend       | Python (ML + logic)            |
| Frontend      | Streamlit / Flask              |
| ML Models     | OpenCV, PyTorch / TensorFlow   |
| Reporting     | PDF / HTML via insights engine |
| Hosting       | Render / Fly.io (Phase 2)      |
| Auth + DB     | Supabase / Firebase (Phase 2)  |

---

## 📌 Notes
- All Phase 1 tasks are structured for **local testing** (no login or cloud).
- Images/videos for testing are provided in `/data/`
- Reports should follow WHO compliance as per YAML and markdown guidelines.

---

Need help? Comment on any issue or reach out directly via GitHub.
