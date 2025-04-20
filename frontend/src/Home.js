import React from 'react';
import { Link } from 'react-router-dom';
import './App.css';



const Home = () => {
  return (
    <div className="form-container">
      <h2>Главное меню</h2>
      <nav>
        <ul style={{ listStyleType: 'none', padding: 0 }}>
          <li style={{ margin: '15px 0' }}>
            <Link to="/add-split" className="menu-link">Добавить распределение прав</Link>
          </li>
          <li style={{ margin: '15px 0' }}>
            <Link to="/get-splits" className="menu-link">Просмотр распределений</Link>
          </li>
          <li style={{ margin: '15px 0' }}>
            <Link to="/delete-splits" className="menu-link">Удалить все распределения</Link>
          </li>
          <li style={{ margin: '15px 0' }}>
            <Link to="/generate-reports" className="menu-link">Генерация отчетов</Link>
          </li>
        </ul>
      </nav>
    </div>
  );
};

export default Home;