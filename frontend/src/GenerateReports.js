import React, { useState } from 'react';
import axios from 'axios';
import './GenerateReports.css';
import BackButton from './BackButton';
import { API_BASE_URL } from './config';

const GenerateReports = () => {
  const [files, setFiles] = useState([]);
  const [status, setStatus] = useState({ type: '', message: '' });
  const [isLoading, setIsLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (files.length === 0) {
      setStatus({ type: 'error', message: 'Выберите хотя бы один файл' });
      return;
    }

    setIsLoading(true);
    setStatus({ type: 'loading', message: 'Генерация отчетов...' });

    try {
      const formData = new FormData();
      files.forEach(file => formData.append('reports', file));

      // Вариант 1: Используем fetch вместо axios для лучшего контроля
      const response = await fetch(`${API_BASE_URL}/generate-reports/`, {
        method: 'POST',
        body: formData,
        credentials: 'include',
        // Не устанавливаем Content-Type вручную для FormData!
      });

      // Обработка ответа
      if (!response.ok) {
        // Пытаемся прочитать ошибку как JSON или текст
        let errorData;
        try {
          errorData = await response.json();
        } catch {
          errorData = { detail: await response.text() };
        }
        throw new Error(errorData.detail || 'Ошибка сервера');
      }

      // Создаем ссылку для скачивания
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', 'reports.zip');
      document.body.appendChild(link);
      link.click();
      link.remove();

      setStatus({ type: 'success', message: 'Отчеты успешно сгенерированы' });
    } catch (error) {
      let errorMessage = 'Ошибка при генерации отчетов';
      
      // Улучшенная обработка ошибок
      if (error.message.includes('Failed to fetch')) {
        errorMessage = 'Ошибка соединения с сервером';
      } else if (error.message) {
        errorMessage = error.message;
      }

      setStatus({ 
        type: 'error', 
        message: errorMessage,
      });
      
      console.error('Error details:', error);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="generate-container">
      <BackButton />
      <h2>Генерация отчетов</h2>
      <form onSubmit={handleSubmit}>
        <div className="form-group">
          <label>Загрузите отчеты (CSV/XLSX):</label>
          <input
            type="file"
            multiple
            onChange={(e) => {
              setFiles([...e.target.files]);
              setStatus({ type: '', message: '' });
            }}
            accept=".csv,.xlsx,.xls"
            disabled={isLoading}
          />
          <div className="file-info">
            {files.length > 0 ? (
              <ul>
                {Array.from(files).map((file, index) => (
                  <li key={index}>{file.name}</li>
                ))}
              </ul>
            ) : (
              'Файлы не выбраны'
            )}
          </div>
        </div>
        
        <button 
          type="submit" 
          className="generate-button"
          disabled={isLoading || files.length === 0}
        >
          {isLoading ? (
            <>
              <span className="spinner"></span>
              Генерация...
            </>
          ) : (
            'Сгенерировать отчеты'
          )}
        </button>
        
        {status.message && (
          <div className={`status-message ${status.type}`}>
            {status.message}
            {status.type === 'error' && (
              <div className="error-details">
                Проверьте:
                <ul>
                  <li>Формат файлов (только CSV/XLSX)</li>
                  <li>Размер файлов (не более 10MB каждый)</li>
                  <li>Соединение с интернетом</li>
                </ul>
              </div>
            )}
          </div>
        )}
      </form>
    </div>
  );
};

export default GenerateReports;