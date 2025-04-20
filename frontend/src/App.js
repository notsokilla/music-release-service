import React from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import Home from './Home';
import AddSplit from './AddSplit';
import GetSplits from './GetSplits';
import DeleteSplits from './DeleteSplits';
import GenerateReports from './GenerateReports';
import './App.css';

const App = () => {
  return (
    <Router>
      <div className="app-container">
        <h1 className="header">Система управления музыкальными отчетами</h1>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/add-split" element={<AddSplit />} />
          <Route path="/get-splits" element={<GetSplits />} />
          <Route path="/delete-splits" element={<DeleteSplits />} />
          <Route path="/generate-reports" element={<GenerateReports />} />
        </Routes>
      </div>
    </Router>
  );
};

export default App;