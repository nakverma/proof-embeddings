# Processing
## Annotation
### Annotation in Windows
The `annotate_windows.py` script will work with already compiled PDF files and flashed them on screen for annotation purposes. It can be called with:
```
python annotate_windows.py --root_dir=<ROOT_DIR> --done_path=<DONE_PATH>
```
* The `<ROOT_DIR>` is the path to the outmost directory containing data files. You can download this from Google Drive from the following path `2019 Spring Discrete/ML Proof Pattern Learning//CTL//Database//data.zip`, and unzip it. This directory is organized in the following manner: `<SEMESTER>/<HW>/<SUBMISSIONS>` (`e.g. F19/HW2/22432_5818111_main-1.tex`). However, for `Windows` distributions, you have to compile the `.tex` files into PDFs and place them in `<SUBMISSIONS>` instead of the `.tex` files.

* The `<DONE_PATH>` is an arbitrary path to a .txt file of your choice (created automatically if it doesn't exist) to keep a list of files that you've annotated.

### Annotation in UNIX
The `annotate.py` script compiles LaTeX submission files into PDFs and flashes them on screen for annotation purposes. It can be called with:
```
python annotate.py --root_dir=<ROOT_DIR> --output_dir=<OUTPUT_DIR> --done_path=<DONE_PATH>
```

* The `<ROOT_DIR>` is the path to the outmost directory containing data files. You can download this from Google Drive from the following path `2019 Spring Discrete/ML Proof Pattern Learning//CTL//Database//data.zip`, and unzip it. This directory is organized in the following manner: `<SEMESTER>/<HW>/<SUBMISSIONS>` (`e.g. F19/HW2/22432_5818111_main-1.tex`).

* The `<OUTPUT_DIR>` is an arbitrary path to a directory of your choice (created automatically if it doesn't exist) to dump the generated PDF and metadata files.

* The `<DONE_PATH>` is an arbitrary path to a .txt file of your choice (created automatically if it doesn't exist) to keep a list of files that you've annotated.

* The Python script calls the `pdflatex` command to compile the .tex files. You can find more information on how to download it from [here](https://linux.die.net/man/1/pdflatex).

### Notes
* The `pdflatex` command is not available in Windows distributions. You can download [Cygwin](https://cygwin.com/install.html) in this case and work from there.
* The `pdflatex` command fails to exit when there is an unexpected syntax in the LaTeX files. The error messages are not printed on the screen. In these cases, you might need to press the ENTER button multiple times until the PDF shows up on the screen.
* When you're finished annotating a submission, you can type `N` or `n` on the console to move on to the next submission. This will update the .txt file specified with the `<DONE_PATH>` argument. On the other hand, you can type `Q` or `q` to exit the program without updating the .txt file.
* When the program is run, it will prompt you to select the semester no. and homework no. that you want to annotate. The options are also listed.