import React from "react";
import { BrowserRouter as Router, Route, Routes } from "react-router-dom";
import Dashboard from "./Dashboard";
import DataPortal from "./DataPortal";
import AiPortal from "./AiPortal";
import DataDisplay from "./DataDisplay";
import LandingPage from "./LandingPage";
import Auth from "./Auth";
import ModelDisplay from "./ModelDisplay";
import Profile from "./Profile";
import ChoosePlan from "./ChoosePlan";
import Verify from "./Verify";

const MyRouter = () => {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/auth" element={<Auth />} />
        <Route path="/dataportal" element={<DataPortal />} />
        <Route path="/aiportal" element={<AiPortal />} />
        <Route path="/data" element={<DataDisplay />} />
        <Route path="/models" element={<ModelDisplay />} />
        <Route path="/landing" element={<LandingPage />} />
        <Route path="/profile" element={<Profile />} />
        <Route path="/chooseplan" element={<ChoosePlan />} />
        <Route path="/verify/:token" element={<Verify />} />
      </Routes>
    </Router>
  );
};

export default MyRouter;
