#!/bin/bash
# Фронтенд
cd frontend
npm install
npm run build

# Копирование статики
cd ..
mkdir -p backend/static
cp -r frontend/build/* backend/static/

# Бекенд
cd backend
if [ -f "requirements.txt" ]; then
  pip install -r requirements.txt
else
  echo "Error: requirements.txt not found!" >&2
  exit 1
fi