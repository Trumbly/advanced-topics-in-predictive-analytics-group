# docs/report/Makefile
.PHONY: docx clean
docx: report.docx
report.docx: report.md
	pandoc report.md -o report.docx
clean:
	rm -f report.docx
