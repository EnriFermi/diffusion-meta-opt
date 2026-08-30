#!/bin/sh
# Установка pyenv, Python 3.13, pipenv и создание окружения для проекта

set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROJECT_ROOT=$(dirname -- "$SCRIPT_DIR")
cd "$PROJECT_ROOT"

PYTHON_VERSION="3.13"
echo "==> Проект: $PROJECT_ROOT"

# --- Установка pyenv ---
if command -v pyenv >/dev/null 2>&1; then
    echo "==> pyenv уже установлен: $(pyenv --version 2>/dev/null || echo unknown)"
else
    echo "==> Установка pyenv..."
    if command -v brew >/dev/null 2>&1; then
        brew install pyenv
    else
        curl -fsSL https://pyenv.run | bash
        echo "==> Добавьте pyenv в PATH (в ~/.bashrc или ~/.zshrc):"
        echo '    export PYENV_ROOT="$HOME/.pyenv"'
        echo '    [ -d "$PYENV_ROOT/bin" ] && export PATH="$PYENV_ROOT/bin:$PATH"'
        echo '    eval "$(pyenv init -)"'
        export PYENV_ROOT="$HOME/.pyenv"
        if [ -d "$PYENV_ROOT/bin" ]; then
            export PATH="$PYENV_ROOT/bin:$PATH"
        fi
        eval "$(pyenv init -)" 2>/dev/null || true
    fi
fi

# Инициализация pyenv в текущей сессии
export PYENV_ROOT="${PYENV_ROOT:-$HOME/.pyenv}"
if [ -d "$PYENV_ROOT/bin" ]; then
    export PATH="$PYENV_ROOT/bin:$PATH"
fi
eval "$(pyenv init -)" 2>/dev/null || true

# --- Установка Python 3.13 ---
INSTALLED_PY=$(pyenv versions --bare 2>/dev/null | grep -E "^${PYTHON_VERSION}[.]" | tail -n 1)
if [ -n "$INSTALLED_PY" ]; then
    echo "==> Python уже установлен: $INSTALLED_PY"
else
    echo "==> Установка Python ${PYTHON_VERSION} (может занять несколько минут)..."
    pyenv install -s "${PYTHON_VERSION}"
    INSTALLED_PY=$(pyenv versions --bare | grep -E "^${PYTHON_VERSION}[.]" | tail -n 1)
fi

# Установить версию для проекта
pyenv local "${INSTALLED_PY:-$PYTHON_VERSION}"

# --- Установка pipenv ---
if command -v pipenv >/dev/null 2>&1; then
    echo "==> pipenv уже установлен: $(pipenv --version 2>/dev/null || echo unknown)"
else
    echo "==> Установка pipenv..."
    if command -v pipx >/dev/null 2>&1; then
        pipx install pipenv
    else
        python -m pip install --user pipenv
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
