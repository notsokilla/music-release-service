import React, { useState } from 'react';
import axios from 'axios';
import './GenerateReports.css';
import BackButton from './BackButton';
import { API_BASE_URL } from './api/config';  // Относительный путь внутри src/

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

      const response = await axios.post(`${API_BASE_URL}/generate-reports`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        responseType: 'blob'
      });

      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', 'reports.zip');
      document.body.appendChild(link);
      link.click();
      link.remove();

      setStatus({ type: 'success', message: 'Отчеты успешно сгенерированы' });
    } catch (error) {
      let errorMessage = 'Ошибка при генерации отчетов';
      
      if (error.response) {
        if (error.response.data instanceof Blob) {
          const text = await error.response.data.text();
          try {
            const json = JSON.parse(text);
            errorMessage = json.detail || errorMessage;
          } catch {
            errorMessage = text || errorMessage;
          }
        } else {
          errorMessage = error.response.data.detail || errorMessage;
        }
      }
      
      setStatus({ 
        type: 'error', 
        message: errorMessage 
      });
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
            {files.length > 0 ? `Выбрано файлов: ${files.length}` : 'Файлы не выбраны'}
          </div>
        </div>
        
        <button 
          type="submit" 
          className="generate-button"
          disabled={isLoading || files.length === 0}
        >
          {isLoading ? 'Генерация...' : 'Сгенерировать отчеты'}
        </button>
        
        {status.message && (
          <div className={`status-message ${status.type}`}>
            {status.message}
            {status.type === 'error' && (
              <div style={{ marginTop: '10px', fontSize: '0.9em' }}>
                Проверьте формат файлов или обратитесь к администратору
              </div>
            )}
          </div>
        )}
      </form>
    </div>
  );
};

export default GenerateReports;