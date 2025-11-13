set PROJECT_DIR=%~dp0
cd /d "%PROJECT_DIR%"

REM Run sphinx build and capture output
echo About to run make command...
call make html SPHINXOPTS="-a" > build_output.log 2>&1
echo Make command finished with exit code: %ERRORLEVEL%

REM Debug: Show what was captured
echo Debug: Contents of build_output.log:
echo =====================================
type build_output.log
echo =====================================

REM Check if there were any WARNING messages in the output
findstr /i "WARNING" build_output.log >nul 2>&1
echo Debug: findstr exit code = %ERRORLEVEL%

if %ERRORLEVEL% equ 0 (
    echo Build completed with WARNINGS. Press any key to continue...
    pause >nul
) else (
    echo Build completed successfully with no warnings. Closing window...
    timeout /t 1 >nul
)

REM Clean up temporary log file
del build_output.log 2>nul