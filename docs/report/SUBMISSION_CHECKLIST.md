# Submission checklist (D4 project report)

The report at `docs/report/report.md` is otherwise complete. The
following items require team action before the 2026-05-20 deadline:

- [ ] **Record the demo video** (D3, max 5 min). Replace the `*TODO*`
  line under the abstract with the unlisted YouTube URL.
- [ ] **§6 individual contributions sign-off.** Each teammate confirms
  or rewrites their own paragraph under `# 6. DRAFT: Individual
  contributions`. Once all four have signed off, remove `DRAFT:` from
  the heading and the sign-off blockquote.
- [ ] **Rebuild the Word doc** after the two edits above:
  ```
  cd docs/report && make docx
  ```
- [ ] **Sanity-check page count** in Word (must be ≤ 10). At final
  verification (2026-05-15) the rebuilt `report.docx` is **11 pages**
  in Word — one over the 10-page budget. The body prose has already
  been trimmed per spec (§1 cut ~120 words, §4 cut ~120 words, §2/§3
  cut ~150 words combined). The remaining 1-page overrun is
  structural — Figure 1 (the agent loop diagram) consumes ~½ of page 2,
  Figure 2 ~½ of page 5, Figure 3 ~½ of page 6, and the §3 table
  splits across pages 6→7. Three options for the team to consider:
  1. **Tighten Word page setup** before final export: reduce body
     font to 10pt, switch heading style spacing to "compact", or
     narrow margins to 0.8 inch. Any one of these likely drops to 10.
  2. **Shrink the figures in Word** (right-click → Size → ~80%) so
     captions sit next to the image instead of below it. Particularly
     helpful for Figure 1.
  3. **Accept 11 pages** if the rubric phrases the limit as "around
     10" rather than a hard cap. We did not cut §5 because the spec
     explicitly says the rubric rewards honesty there.
- [ ] **Submit** the resulting `docs/report/report.docx` per course
  instructions (alongside the Kaggle submissions and the GitHub repo
  link from the README).
