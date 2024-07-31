import React, { useState, useEffect } from "react";
import Spline from "@splinetool/react-spline";
import Popup from "./Popup";
import InsertRowForm from "./InsertRowForm";
import PredictionForm from "./PredictionForm";
import StoreModel from "./StoreModel";
import AddFeatureForm from "./AddFeatureForm";
import { useNavigate } from "react-router-dom";
import { useMediaQuery } from "react-responsive";

import { Box, IconButton } from "@mui/material";
import AddIcon from "@mui/icons-material/Add";
import AppsIcon from "@mui/icons-material/Apps";
import HelpIcon from "@mui/icons-material/Help";
import ArchiveIcon from "@mui/icons-material/Archive";
import PersonIcon from "@mui/icons-material/Person";
import BoltIcon from "@mui/icons-material/Bolt";

const Dashboard = () => {
  const [isInsertPopupOpen, setIsInsertPopupOpen] = useState(false);
  const [isPredictionPopupOpen, setIsPredictionPopupOpen] = useState(false);
  const [isStoreModelPopupOpen, setIsStoreModelPopupOpen] = useState(false);
  const [isAddFeaturePopupOpen, setIsAddFeaturePopupOpen] = useState(false);
  const [recentData, setRecentData] = useState([]);
  const [recentModels, setRecentModels] = useState([]);
  const [loading, setLoading] = useState(false);
  const token = localStorage.getItem("token");
  const navigate = useNavigate();
  const isSmallScreen = useMediaQuery({ maxWidth: 768 });

  const handleNavigation = (path) => {
    navigate(path);
  };

  useEffect(() => {
    const fetchRecentData = async () => {
      try {
        const response = await fetch("/api/recent_data/", {
          method: "GET",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${token}`, // Include JWT in headers
          },
        });
        const data = await response.json();
        if (data.auth) {
          navigate("/auth");
        }
        if (data.recent_data) {
          setRecentData(data.recent_data);
        }
      } catch (error) {
        console.error("Error fetching recent data:", error);
      }
    };

    fetchRecentData();
  }, []);

  useEffect(() => {
    const fetchRecentModels = async () => {
      try {
        setLoading(true);
        const response = await fetch("/api/recent_models/", {
          method: "GET",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${token}`, // Include JWT in headers
          },
        });
        const data = await response.json();
        if (data.auth) {
          navigate("/auth");
        }
        if (data.recent_models) {
          setRecentModels(data.recent_models);
        }
        setLoading(false);
      } catch (error) {
        console.error("Error fetching recent models:", error);
      }
    };

    fetchRecentModels();
  }, []);

  useEffect(() => {
    if (!token) {
      navigate("/auth");
    }
  }, []);

  const openInsertPopup = () => setIsInsertPopupOpen(true);
  const closeInsertPopup = () => setIsInsertPopupOpen(false);

  const openPredictionPopup = () => setIsPredictionPopupOpen(true);
  const closePredictionPopup = () => setIsPredictionPopupOpen(false);

  const openStoreModelPopup = () => setIsStoreModelPopupOpen(true);
  const closeStoreModelPopup = () => setIsStoreModelPopupOpen(false);

  const openAddFeaturePopup = () => setIsAddFeaturePopupOpen(true);
  const closeAddFeaturePopup = () => setIsAddFeaturePopupOpen(false);

  if (isSmallScreen) {
    return (
      <div className="flex flex-col items-center justify-center h-screen">
        <img src="../../static/images/SaleSightLogo.png" className="w-1/4" />
        <h1 className="text-5xl text-center font-reem-kufi pt-6 pb-5">
          Not configured for mobile
        </h1>
        <p className="text-xl font-reem-kufi">
          Please use SaleSight on a computer
        </p>
      </div>
    );
  }

  return (
    <div className="h-screen bg-white text-gray-900 p-5 flex flex-col px-20">
      {/* Main Content */}
      <div className="flex-grow flex pt-8 h-64">
        {/* Left Section */}
        <div className="flex flex-col justify-start w-2/5 space-y-4">
          <h1 className="text-3xl font-abo-one pb-6">My Dashboard</h1>
          <div className="items-start flex flex-row space-x-12 justify-start">
            <button
              onClick={() => handleNavigation("/dataportal")}
              className="px-7 py-4 bg-mygreen hover:bg-lightgreen transition duration-200 text-black border-none rounded-2xl text-xl font-reem-kufi cursor-pointer"
            >
              Data Portal
            </button>
            <button
              onClick={() => handleNavigation("/aiportal")}
              className="px-11 py-4 bg-cyan-300 hover:bg-cyan-200 transition duration-200 text-black border-none rounded-2xl text-xl font-reem-kufi cursor-pointer"
            >
              Ai Portal
            </button>
          </div>

          <div className="flex flex-col items-start space-y-8 pt-8">
            <h1 className="font-abo-one text-2xl">Quick Access Buttons</h1>
            <Box className="flex flex-row space-x-8">
              <IconButton
                onClick={() => setIsInsertPopupOpen(!isInsertPopupOpen)}
                sx={{
                  backgroundColor: "#B2FF97",
                  borderRadius: "50%",
                  p: 1.5,
                }}
              >
                <AddIcon fontSize="large" />
              </IconButton>
              <IconButton
                onClick={() => setIsAddFeaturePopupOpen(!isAddFeaturePopupOpen)}
                sx={{ backgroundColor: "#B2FF97", borderRadius: "50%", p: 1.5 }}
              >
                <AppsIcon fontSize="large" />
              </IconButton>
              <IconButton
                onClick={() => setIsPredictionPopupOpen(!isPredictionPopupOpen)}
                sx={{
                  backgroundColor: "#9CEAFB",
                  borderRadius: "50%",
                  p: 1.5,
                }}
              >
                <HelpIcon fontSize="large" />
              </IconButton>
              <IconButton
                onClick={() => setIsStoreModelPopupOpen(!isStoreModelPopupOpen)}
                sx={{ backgroundColor: "#9CEAFB", borderRadius: "50%", p: 1.5 }}
              >
                <ArchiveIcon fontSize="large" />
              </IconButton>
              <IconButton
                onClick={() => handleNavigation("/profile")}
                sx={{ backgroundColor: "#FCFF60", borderRadius: "50%", p: 1.5 }}
              >
                <PersonIcon fontSize="large" />
              </IconButton>
              <IconButton
                onClick={() => handleNavigation("/chooseplan")}
                sx={{ backgroundColor: "#FCFF60", borderRadius: "50%", p: 1.5 }}
              >
                <BoltIcon fontSize="large" />
              </IconButton>
            </Box>
          </div>
        </div>

        {/* Middle Section */}
        <div className="flex flex-col items-center justify-start w-1/5 space-y-4">
          <img src="../../static/images/SaleSightLogo.png" className="w-1/3" />
        </div>

        {/* Right Section */}
        <div className="flex flex-col pl-12 w-2/5 space-y-4 h-full justify-between">
          <div className="justify-start">
            <div className="flex flex-row items-baseline">
              <h2 className="text-4xl font-abo-one text-cyan-500">My Models</h2>
              <a
                href="/models"
                className="hover:underline text-sm ml-12 text-gray-500"
              >
                <span className="font-reem-kufi text-lg">View all -&gt;</span>
              </a>
            </div>
            <div className="flex justify-between pt-4">
              {loading ? (
                <div className="flex w-full pl-16 items-center justify-start">
                  <svg
                    class="pl"
                    width="240"
                    height="240"
                    viewBox="0 0 240 240"
                  >
                    <circle
                      class="pl__ring pl__ring--a"
                      cx="120"
                      cy="120"
                      r="105"
                      fill="none"
                      stroke="#000"
                      stroke-width="20"
                      stroke-dasharray="0 660"
                      stroke-dashoffset="-330"
                      stroke-linecap="round"
                    ></circle>
                    <circle
                      class="pl__ring pl__ring--b"
                      cx="120"
                      cy="120"
                      r="35"
                      fill="none"
                      stroke="#000"
                      stroke-width="20"
                      stroke-dasharray="0 220"
                      stroke-dashoffset="-110"
                      stroke-linecap="round"
                    ></circle>
                    <circle
                      class="pl__ring pl__ring--c"
                      cx="85"
                      cy="120"
                      r="70"
                      fill="none"
                      stroke="#000"
                      stroke-width="20"
                      stroke-dasharray="0 440"
                      stroke-linecap="round"
                    ></circle>
                    <circle
                      class="pl__ring pl__ring--d"
                      cx="155"
                      cy="120"
                      r="70"
                      fill="none"
                      stroke="#000"
                      stroke-width="20"
                      stroke-dasharray="0 440"
                      stroke-linecap="round"
                    ></circle>
                  </svg>
                </div>
              ) : (
                <>
                  {recentModels.map((model, index) => (
                    <div
                      key={index}
                      onClick={openPredictionPopup}
                      className="bg-white cursor-pointer w-1/2 pb-4 rounded-lg flex flex-col justify-start items-center"
                    >
                      <div className="justify-center w-3/5 items-center bg-white relative">
                        <img
                          src="../../static/images/AiIcon.png"
                          className=""
                        />
                      </div>
                      <div className="text-2xl pt-2 font-reem-kufi text-cyan-600">
                        {model.name}
                      </div>
                      <div className="font-reem-kufi text-gray-500">
                        {model.created_at}
                      </div>
                    </div>
                  ))}
                </>
              )}
            </div>
          </div>
          <div className="flex flex-row items-baseline">
            <h2 className="text-4xl font-abo-one text-brightgreen">
              Recent Data
            </h2>
            <a
              href="/data"
              className="hover:underline text-sm ml-12 text-gray-500"
            >
              <span className="font-reem-kufi text-lg">View all -&gt;</span>
            </a>
          </div>
        </div>
      </div>

      {/* Recent Data Section */}
      <div className="flex flex-col pt-4 font-reem-kufi">
        <div className="space-y-4">
          <table className="table-fixed w-full">
            <thead>
              <tr className="bg-white h-12 rounded-xl w-full justify-between items-center">
                <th className="text-center font-semibold border-4 border-gray-400 py-2 w-full">
                  Response
                </th>
                <th className="text-center font-semibold border-4 border-gray-400 py-2 w-full">
                  Location
                </th>
                <th className="text-center font-semibold border-4 border-gray-400 py-2 w-full">
                  Offer
                </th>
                <th className="text-center font-semibold border-4 border-gray-400 py-2 w-full">
                  Sale Product
                </th>
                <th className="text-center font-semibold border-4 border-gray-400 py-2 w-full">
                  Lead Gen
                </th>
                <th className="text-center font-semibold border-4 border-gray-400 py-2 w-full">
                  Funnel Stage
                </th>
                <th className="text-center font-semibold border-4 border-gray-400 py-2 w-full">
                  Result
                </th>
              </tr>
            </thead>
            <tbody>
              {loading ? (
                <tr>
                  <td />
                  <td />
                  <td />
                  <td className="flex justify-center items-center h-24">
                    <svg className="pl" viewBox="0 0 240 240">
                      <circle
                        className="pl__ring pl__ring--a"
                        cx="120"
                        cy="120"
                        r="105"
                        fill="none"
                        stroke="#000"
                        strokeWidth="20"
                        strokeDasharray="0 660"
                        strokeDashoffset="-330"
                        strokeLinecap="round"
                      ></circle>
                      <circle
                        className="pl__ring pl__ring--b"
                        cx="120"
                        cy="120"
                        r="35"
                        fill="none"
                        stroke="#000"
                        strokeWidth="20"
                        strokeDasharray="0 220"
                        strokeDashoffset="-110"
                        strokeLinecap="round"
                      ></circle>
                      <circle
                        className="pl__ring pl__ring--c"
                        cx="85"
                        cy="120"
                        r="70"
                        fill="none"
                        stroke="#000"
                        strokeWidth="20"
                        strokeDasharray="0 440"
                        strokeLinecap="round"
                      ></circle>
                      <circle
                        className="pl__ring pl__ring--d"
                        cx="155"
                        cy="120"
                        r="70"
                        fill="none"
                        stroke="#000"
                        strokeWidth="20"
                        strokeDasharray="0 440"
                        strokeLinecap="round"
                      ></circle>
                    </svg>
                  </td>
                  <td />
                  <td />
                  <td />
                </tr>
              ) : (
                <>
                  {recentData.map((row, index) => (
                    <tr
                      key={index}
                      className="bg-white h-12 rounded w-full justify-between px-8 items-center font-reem-kufi"
                    >
                      <td className="text-center border-2 border-gray-400">
                        {row.response}
                      </td>
                      <td className="text-center border-2 border-gray-400">
                        {row.location}
                      </td>
                      <td className="text-center border-2 border-gray-400">
                        {row.offer}
                      </td>
                      <td className="text-center border-2 border-gray-400">
                        {row.sale_product}
                      </td>
                      <td className="text-center border-2 border-gray-400">
                        {row.lead_gen}
                      </td>
                      <td className="text-center border-2 border-gray-400">
                        {row.funnel_stage}
                      </td>
                      <td className="text-center border-2 border-gray-400">
                        {row.result}
                      </td>
                    </tr>
                  ))}
                </>
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Popups */}
      <Popup isOpen={isInsertPopupOpen} onClose={closeInsertPopup}>
        <InsertRowForm />
      </Popup>
      <Popup isOpen={isPredictionPopupOpen} onClose={closePredictionPopup}>
        <PredictionForm />
      </Popup>
      <Popup isOpen={isStoreModelPopupOpen} onClose={closeStoreModelPopup}>
        <StoreModel />
      </Popup>
      <Popup isOpen={isAddFeaturePopupOpen} onClose={closeAddFeaturePopup}>
        <AddFeatureForm />
      </Popup>
    </div>
  );
};

export default Dashboard;
