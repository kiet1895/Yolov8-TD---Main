@echo off
echo =====================================
echo Dang dong goi ung dung thanh EXE...
echo =====================================
echo.

REM Kich hoat moi truong ao
if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
) else (
    echo CANH BAO: Khong tim thay moi truong ao .venv
)

REM Cai dat PyInstaller neu chua co
echo Dang kiem tra PyInstaller...
pip show pyinstaller >nul 2>&1
if errorlevel 1 (
    echo Dang cai dat PyInstaller...
    pip install pyinstaller
)

REM Xoa cac file build cu
echo Dang xoa cac file build cu...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist

REM Build exe
echo.
echo Dang build ung dung...
pyinstaller --clean build.spec

if errorlevel 1 (
    echo.
    echo =====================================
    echo LOI: Qua trinh build that bai!
    echo =====================================
    pause
    exit /b 1
)

echo.
echo =====================================
echo THANH CONG! File EXE da duoc tao tai:
echo dist\YoloV8_TD\YoloV8_TD.exe
echo =====================================
echo.

REM Mo thu muc dist
explorer dist\YoloV8_TD

pause
