chcp 65001 >nul


@echo off

echo Активация venv...
call venv\Scripts\activate

echo Установка зависимостей...
call pip install -r requirements.txt

echo Проверка пакетов...
call pip list