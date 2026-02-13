#!/usr/bin/env bash
# Установка pipenv и создание окружения для проекта

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "==> Проект: $PROJECT_ROOT"

# --- Установка pipenv ---
if command -v pipenv &>/dev/null; then
    echo "==> pipenv уже установлен: $(pipenv --version)"
else
    echo "==> Установка pipenv..."
    if command -v pipx &>/dev/null; then
        pipx install pipenv
    else
        pip install --user pipenv
        echo "==> Добавьте ~/.local/bin в PATH, если pipenv не находится"
        export PATH="$HOME/.local/bin:$PATH"
    fi
fi

# --- Создание окружения ---
echo "==> Создание окружения (pipenv install)..."
pipenv install --dev

echo ""
echo "==> Готово. Активируйте окружение:"
echo "    pipenv shell"
echo ""
echo "Или запускайте команды через pipenv run:"
echo "    pipenv run python your_script.py"
