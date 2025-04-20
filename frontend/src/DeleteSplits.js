import React, { useState } from 'react';
import axios from 'axios';
import BackButton from './BackButton';
import { API_BASE_URL } from './config';  // Относительный путь внутри src/



const DeleteSplits = () => {
  const [status, setStatus] = useState({ type: '', message: '' });
  const [isLoading, setIsLoading] = useState(false);

  const handleDelete = async () => {
    if (!window.confirm('Вы уверены, что хотите удалить ВСЕ распределения прав?')) {
      return;
    }

    setIsLoading(true);
    setStatus({ type: 'loading', message: 'Удаление распределений...' });

    try {
      const response = await axios.delete(`${API_BASE_URL}/delete-splits`);
      setStatus({ type: 'success', message: response.data.message });
    } catch (error) {
      const errorMessage = error.response?.data?.detail || 'Ошибка при удалении распределений';
      setStatus({ type: 'error', message: errorMessage });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="form-container">
      <BackButton />
      <h2>Удалить все распределения прав</h2>
      <button 
        onClick={handleDelete}
        disabled={isLoading}
        className="delete-button"
      >
        {isLoading ? 'Удаление...' : 'Удалить все распределения'}
      </button>
      
      {status.message && (
        <div className={`status-message ${status.type}`}>
          {status.message}
        </div>
      )}
    </div>
  );
};

export default DeleteSplits;