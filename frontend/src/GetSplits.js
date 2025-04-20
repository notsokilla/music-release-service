import React, { useState, useEffect } from 'react';
import axios from 'axios';
import BackButton from './BackButton';
import { API_BASE_URL } from '../api/config';

const GetSplits = () => {
  const [splits, setSplits] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchSplits = async () => {
      try {
        const response = await axios.get(`${API_BASE_URL}/get-splits`);
        setSplits(response.data);
      } catch (err) {
        const errorMessage = err.response?.data?.detail || 'Не удалось загрузить распределения';
        setError(errorMessage);
      } finally {
        setLoading(false);
      }
    };

    fetchSplits();
  }, []);

  if (loading) return <div className="status-message loading">Загрузка распределений...</div>;
  
  if (error) return (
    <div className="form-container">
      <BackButton />
      <div className="status-message error">
        {error}
        <div style={{ marginTop: '10px', fontSize: '0.9em' }}>
          Попробуйте обновить страницу или обратитесь к администратору
        </div>
      </div>
    </div>
  );

  return (
    <div className="form-container">
      <BackButton />
      <h2>Список распределений прав</h2>
      {splits.length === 0 ? (
        <p>Нет сохраненных распределений</p>
      ) : (
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', marginTop: '20px' }}>
            <thead>
              <tr style={{ backgroundColor: '#f2f2f2' }}>
                <th style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'left' }}>Трек</th>
                <th style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'left' }}>Артист</th>
                <th style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'left' }}>Доля (%)</th>
              </tr>
            </thead>
            <tbody>
              {splits.map((split, index) => (
                <tr key={index} style={{ borderBottom: '1px solid #ddd' }}>
                  <td style={{ padding: '12px', border: '1px solid #ddd' }}>{split.track_title}</td>
                  <td style={{ padding: '12px', border: '1px solid #ddd' }}>{split.artist}</td>
                  <td style={{ padding: '12px', border: '1px solid #ddd' }}>{split.percentage}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
};

export default GetSplits;