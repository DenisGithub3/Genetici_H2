@echo off
setlocal

rem Output file
set output_file=rezultate_Schwefel3.txt

rem Clear the output file if it already exists
> %output_file% echo Results:

rem Loop to run the Python script x times
for /L %%i in (1,1,5) do (
    rem Call the Python script and redirect output to the file
    python Schwefel.py >> %output_file%
)

endlocal