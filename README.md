# CIPOC

This repository stores the code for the Cancer Identification and Precision Oncology Center (CIPOC) at the University of North Carolina at Chapel Hill.

## Purpose

One of the aims of CIPOC is to rapidly identify and characterize cancer cases from structured EHR and free-text clinical notes.

Currently, this codebase is designed for cancer registry variable extraction from clinical notes using LLMs.

## Additional Notes

- This content is very much under development
- All code is being developed in a secure Azure Trusted Research environment (UNC Health SHIRE) which is airgapped with no direct downloads, meaning the code cannot be directly synced and must be manually updated outside of the SHIRE before syncing changes. As a result, this repository may not represent the current state of the code. We will do our best to sync changes as frequently as possible and faithfully recreate branching/changes as they appear in the internal SHIRE repository.
- LLM work is performed in Azure Databricks, so there may be Databricks-specific components. Ultimately these tools will be cloud/system agnostic, but some Databricks-specific features may exist in the meantime (usually depending on the footprint of any refactors required to remove them).
