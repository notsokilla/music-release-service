import React, { useState } from 'react';
import axios from 'axios';
import BackButton from './BackButton';
import { API_BASE_URL } from './config';  // Относительный путь внутри src/


const AddSplit = () => {
  const [formData, setFormData] = useState({
    track_title: '',
    splits: [{ nickname: '', percentage: '' }]
  });
  const [status, setStatus] = useState({ type: '', message: '' });
  const [isLoading, setIsLoading] = useState(false);

  const handleTrackChange = (e) => {
    setFormData({ ...formData, track_title: e.target.value });
  };

  const handleSplitChange = (index, field, value) => {
    const newSplits = [...formData.splits];
    newSplits[index][field] = field === 'percentage' ? parseFloat(value) || 0 : value;
    setFormData({ ...formData, splits: newSplits });
  };

  const addSplit = () => {
    setFormData({
      ...formData,
      splits: [...formData.splits, { nickname: '', percentage: '' }]
    });
  };

  const removeSplit = (index) => {
    const newSplits = [...formData.splits];
    newSplits.splice(index, 1);
    setFormData({ ...formData, splits: newSplits });
  };

  const validateForm = () => {
    if (!formData.track_title.trim()) {
      setStatus({ type: 'error', message: 'Введите название трека' });
      return false;
    }

    const totalPercent = formData.splits.reduce((sum, split) => {
      return sum + (parseFloat(split.percentage) || 0);
    }, 0);

    if (totalPercent > 100) {
      setStatus({ type: 'error', message: `Сумма процентов (${totalPercent}%) превышает 100%` });
      return false;
    }

    return true;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setStatus({ type: '', message: '' });

    if (!validateForm()) return;

    setIsLoading(true);
    setStatus({ type: 'loading', message: 'Сохранение распределения...' });

    try {
      const payload = {
        track_title: formData.track_title,
        splits: formData.splits
          .filter(s => s.nickname.trim() && s.percentage > 0)
          .map(s => ({
            nickname: s.nickname.trim(),
            percentage: parseFloat(s.percentage)
          }))
      };

      const response = await axios.post(`${API_BASE_URL}/add-split`, payload);
      
      setStatus({ type: 'success', message: response.data.message });
      setFormData({
        track_title: '',
        splits: [{ nickname: '', percentage: '' }]
      });
    } catch (error) {
      const errorMessage = error.response?.data?.detail || 'Ошибка при сохранении распределения';
      setStatus({ type: 'error', message: errorMessage });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="form-container">
      <BackButton />
      <h2>Добавить распределение прав</h2>
      <form onSubmit={handleSubmit}>
        <div className="form-group">
          <label>Название трека:</label>
          <input
            type="text"
            value={formData.track_title}
            onChange={handleTrackChange}
            required
            disabled={isLoading}
          />
        </div>

        <h3>Распределение долей:</h3>
        {formData.splits.map((split, index) => (
          <div key={index} className="form-group">
            <label>Артист:</label>
            <input
              type="text"
              value={split.nickname}
              onChange={(e) => handleSplitChange(index, 'nickname', e.target.value)}
              required
              disabled={isLoading}
            />
            <label>Доля (%):</label>
            <input
              type="number"
              min="0"
              max="100"
              step="0.01"
              value={split.percentage}
              onChange={(e) => handleSplitChange(index, 'percentage', e.target.value)}
              required
              disabled={isLoading}
            />
            {formData.splits.length > 1 && (
              <button 
                type="button" 
                onClick={() => removeSplit(index)}
                disabled={isLoading}
                className="remove-button"
              >
                Удалить
              </button>
            )}
          </div>
        ))}

        <div style={{ margin: '15px 0' }}>
          <button 
            type="button" 
            onClick={addSplit}
            disabled={isLoading}
            className="add-button"
          >
            Добавить артиста
          </button>
        </div>

        <button 
          type="submit" 
          disabled={isLoading}
          className="submit-button"
        >
          {isLoading ? 'Сохранение...' : 'Сохранить распределение'}
        </button>

        {status.message && (
          <div className={`status-message ${status.type}`}>
            {status.message}
            {status.type === 'error' && (
              <div style={{ marginTop: '10px', fontSize: '0.9em' }}>
                Проверьте введенные данные и попробуйте снова
              </div>
            )}
          </div>
        )}
      </form>
    </div>
  );
};

export default AddSplit;