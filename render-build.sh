#!/bin/bash
# Установка фронтенда
cd frontend
npm install
npm run build

# Копирование статики в бекенд
cd ..
mkdir -p backend/static
cp -r frontend/build/* backend/static/

# Установка бекенда
cd backend
pip install -r requirements.txt