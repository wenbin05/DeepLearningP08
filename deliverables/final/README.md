# Group 04 final report and video materials

## Start here

- `Group04_Final_Report.pdf`: eight pages of main content plus two pages of references/appendices.
- `Group04_Slides.pdf`: 16 speaking slides and two reference slides, widescreen.
- `Group04_Slides.pptx`: editable PowerPoint with narration in speaker notes.
- `VIDEO_SCRIPT.md` / `Group04_Video_Script.pdf`: timed two-speaker narration and recording instructions.
- `REHEARSAL_QA.md`: likely questions and evidence-bounded answers.
- `evidence/`: unrounded result tables, protocols, readiness, raw anonymized survey responses and the source-grounded failure example.
- `WRITING_AUDIT.md` and `WRITING_CHANGELOG.md`: academic-humanizer audit and changes.

The main narrative is a new final report based on completed experiments, not a relabelled midterm. Results are actual executed measurements. The interpretation and narration are AI-assisted drafts for the authors to review and own.

## Author review before submission

1. Read and understand the results and rewrite the interpretation in your own voice where appropriate. The capstone guide explicitly says: “Write your own analysis, interpretation, and conclusions.” It also requires disclosure and primary-source verification. The guide also says to implement and run experiments yourselves. Since Codex launched the final local jobs, disclose that workflow and clarify its acceptability with the instructor rather than claiming unaided execution. The disclosure accurately states that Codex assisted with implementation, local experiment orchestration, verification and drafting; do not replace it with a claim of unaided work.
2. Confirm the prior 50/50 declaration and actual responsibilities. The suggested speaking split is Wen Bin on slides 1–8 and Junn Rong on slides 9–16, ten minutes each. Each must appear on camera and understand the whole project.
3. Verify bibliography entries against primary sources, particularly WeightLoRA (2026), whose metadata was retained from the supplied midterm. Publisher/arXiv retrieval was blocked in this environment; no independent primary-source recheck is claimed. The report does not reuse the proposal's external benchmark numbers.
4. Confirm that this CPU subset scope and the full-FT exception are clearly understood. It is not the original larger-data plan, full-benchmark reproduction or equal-FLOP comparison. The completed experiment matrix alone does not establish every rubric condition was met.
5. Rehearse for 20 minutes. The accepted range is 18–22 minutes, and content after 22:00 is not assessed. Record clear audio and slides at 720p or higher, with both speakers visible during their parts. Submit `Group04_Presentation.mp4` and `Group04_Slides.pdf`.

The MP4 is intentionally not manufactured: the handout requires the actual members to present and appear on camera. No video has been recorded or submitted by this assistant. These local documents have not been published to GitHub or onQ by this writing pass.

## Build and edit

The report uses the course's article-style presentation, with eight main pages. LaTeX files are editable. From this directory:

```bash
pdflatex -interaction=nonstopmode -halt-on-error Group04_Final_Report.tex
pdflatex -interaction=nonstopmode -halt-on-error Group04_Final_Report.tex
pdflatex -interaction=nonstopmode -halt-on-error Group04_Slides.tex
pdflatex -interaction=nonstopmode -halt-on-error Group04_Slides.tex
```

Edit the PowerPoint directly, or edit `presentation_content.json` and use `export_presentation.py` to regenerate the deck. The exporter needs `python-pptx` and Pillow in a separate document environment; avoid modifying the pinned training environment. The PDF comes from the Beamer source, while the PowerPoint uses editable native text and embedded figures. Both contain the same main content, with different layout engines.

```bash
python export_presentation.py
pandoc VIDEO_SCRIPT.md -o Group04_Video_Script.pdf --pdf-engine=xelatex \
  -V mainfont='DejaVu Sans' -V geometry:margin=0.8in -V fontsize=10pt
```

`build_report_figures.py` regenerates the result figures from the compact evidence JSON. Rounded display values are derived from full-precision saved data. The report's table source is preserved in the `.tex` file.

## Reproduce the CPU matrix

After installing the repository's pinned experiment dependencies in a dedicated environment, run from the repository root:

```bash
bash deliverables/final/reproduce_cpu.sh
```

This is an actual multi-day training command, not a smoke test. It creates separate fresh `results/cpu_reproduction_v1` and `results/cpu_full_ft_reproduction_v1` folders, preserves the full-FT microbatch exception, and runs a complete rescoring check. The 8 GiB limit was the original runtime constraint; a less constrained machine is recommended. Downloads and adequate disk space are required. Hardware and library differences can change exact results. The new reproduction shell wrapper was syntax-checked, not rerun for days during document preparation.

To recheck the existing original artifacts locally:

```bash
python deliverables/final/verify_cpu_results.py
```

This requires the complete original results folders, including model/adapter artifacts, and the pinned evaluation dependencies. The lightweight evidence package alone cannot pass checks that require checkpoints. The existing final check already verified all 39 runs and rescored all 78 test evaluations. The wrapper's default output updates the local `all_checks` summary, not raw predictions or ratings.

## Evidence boundaries

The final source snapshot remains `results/cpu_placement_v1/source_snapshot.zip`. Writing and exporting documents do not alter the frozen training source. Model checkpoints are several GB and are excluded from the compact handoff archive; the full source-and-evidence archive includes predictions, metrics, configurations and code, but not model weights. The main and full-FT protocols are deliberately separate. The 200 survey judgments share five articles and twenty outputs and are not 200 independent observations.
