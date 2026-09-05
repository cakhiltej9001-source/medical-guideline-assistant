# Source Integrity Log

## 2026-09-02: Rejected mislabeled dengue source

- Intended document: National Guidelines for Clinical Management of Dengue Fever.
- URL supplied by the original Clinical Establishments page: `https://clinicalestablishments.mohfw.gov.in/sites/default/files/standard-treatment-guidelines/7051.pdf`.
- Downloaded SHA-256: `d5ce5698af35c5f9d6e3653aff504fc814e4b5e21f34b2d907e7747c925e5666`.
- Actual document found by cover-page and content inspection: *National Roadmap for Kala-azar Elimination in India*, August 2014.
- Decision: reject from the dengue corpus and retain the downloaded artifact only in local quarantine for provenance review.
- Replacement source: NCVBDC/MOHFW *National Guidelines for Clinical Management of Dengue Fever 2023*.

This incident demonstrates why content validation and visual PDF review are required even when a link is published on an official government page.

## 2026-09-06: Manifest governance update

- Manifest version 2 records the selected edition and active-version status for
  every indexed document.
- Publisher license/redistribution terms were not explicitly stated on the source
  pages reviewed, so each entry is marked for license verification rather than
  assuming permission.
- The hybrid SQLite index was rebuilt from the audited 443 chunks with 443 cached
  embedding hits and zero new embedding API calls.
