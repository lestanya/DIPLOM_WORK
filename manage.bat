@echo off
chcp 65001 >nul
title ЖКХ Бот - Менеджер Проекта

REM Сохраняем текущую папку (корневую)
pushd

:menu
color 0B
cls
echo.
echo ========================================
echo       ^| ЖКХ Бот: Менеджер Запуска ^|
echo ========================================
echo.
echo [1] ^| Запустить ВСЁ (VK Бот + Flask)
echo [2] ^| Запустить только VK Бота
echo [3] ^| Запустить только Flask приложение
echo.
echo [Q] ^| Выход
echo.
set /p "choice=Выберите действие: "

if "%choice%"=="" goto menu

REM Восстанавливаем стандартный цвет консоли
color 07

if "%choice%"=="1" (
    cls
    echo [ ^| ЗАПУСК ВСЕХ СЕРВИСОВ ^| ]
    echo.
    echo Запуск VK Бота в новом окне...
    start "VK Бот" cmd /k "python app/bot/vk_bot.py"
    
    echo Запуск Flask приложения в новом окне...
    start "Flask Web App" cmd /k "flask --app flask_app/app run"
    
    echo.
    echo Все сервисы запущены в отдельных окнах.
    pause
    goto menu
)

if "%choice%"=="2" (
    cls
    echo [ ^| ЗАПУСК VK БОТА ^| ]
    echo.
    echo Бот запускается в новом окне...
    start "VK Бот" cmd /k "python app/bot/vk_bot.py"
    goto menu
)

if "%choice%"=="3" (
    cls
    echo [ ^| ЗАПУСК FLASK ПРИЛОЖЕНИЯ ^| ]
    echo.
    echo Flask запускается в новом окне...
    start "Flask Web App" cmd /k "flask --app flask_app/app run"
    goto menu
)

if /i "%choice%"=="q" goto quit

echo Неверный выбор!
pause
goto menu

:quit
color 07
popd
cls
echo До свидания!
timeout /t 2 >nul
exit /b
