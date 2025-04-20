#!/bin/bash
# Фронтенд
cd frontend
npm install
npm run build

# Подготовка статики
cd ..
mkdir -p backend/static
cp -r frontend/build/* backend/static/

# Бекенд
cd backend
pip install -r requirements.txt