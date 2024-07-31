import React, { useState, useEffect } from "react";
import EyeIcon from "@mui/icons-material/RemoveRedEye";
import UploadIcon from "@mui/icons-material/CloudUpload";
import DatabaseIcon from "@mui/icons-material/Storage";
import AssignmentTurnedInIcon from "@mui/icons-material/AssignmentTurnedIn";
import GridViewIcon from "@mui/icons-material/GridView";
import { useNavigate } from "react-router-dom";
import Popup from "./Popup";
import InsertRowForm from "./InsertRowForm";
import PredictionForm from "./PredictionForm";
import StoreModel from "./StoreModel";
import { useMediaQuery } from "react-responsive";

const AiPortal = () => {
  const [isInsertPopupOpen, setIsInsertPopupOpen] = useState(false);
  const [isPredictionPopupOpen, setIsPredictionPopupOpen] = useState(false);
  const [isStoreModelPopupOpen, setIsStoreModelPopupOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [model, setModel] = useState("");
  const isSmallScreen = useMediaQuery({ maxWidth: 768 });

  const openInsertPopup = () => setIsInsertPopupOpen(true);
  const closeInsertPopup = () => setIsInsertPopupOpen(false);

  const openPredictionPopup = () => setIsPredictionPopupOpen(true);
  const closePredictionPopup = () => setIsPredictionPopupOpen(false);

  const openStoreModelPopup = () => setIsStoreModelPopupOpen(true);
  const closeStoreModelPopup = () => setIsStoreModelPopupOpen(false);

  const [models, setModels] = useState([]);
  const navigate = useNavigate();
  const token = localStorage.getItem("token");

  const handleNavigation = (path) => {
    navigate(path);
  };

  useEffect(() => {
    const fetchModels = async () => {
      try {
        setLoading(true);
        const response = await fetch("/api/models/", {
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
        setModels(data.models.slice(0, 4) || []);
        setModel(data.models[0]);
        setLoading(false);
      } catch (error) {
        console.error("Error fetching models:", error);
      }
    };

    fetchModels();
  }, [token]);

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
    <div className="min-h-screen pt-8 bg-white text-center p-4">
      {/* Top Section */}
      <div
        className="flex justify-between items-center pb-4"
        style={{ height: "20%" }}
      >
        <div className="w-1/3 ">
          <h1 className="text-4xl ml-12 text-left font-abo-one text-cyan-600">
            Ai Portal
          </h1>
        </div>
        <div className="w-1/3 justify-center items-center">
          <img
            src="../../static/images/SaleSightLogo.png"
            alt="Eye"
            className="w-1/6 mx-auto"
          />
        </div>
        <div className="space-x-8 w-1/3 justify-end">
          <button
            onClick={() => handleNavigation("/dataportal")}
            className="bg-mygreen hover:bg-lightgreen rounded-xl font-abo-one py-3 px-6 text-black transition duration-200"
          >
            Data Portal
          </button>
          <button
            onClick={() => handleNavigation("/dashboard")}
            className="bg-yellow-300 hover:bg-yellow-200 py-3 px-6 rounded-xl font-abo-one text-black transition duration-200"
          >
            Home
          </button>
        </div>
      </div>
      {/* Middle Section */}
      <div className="flex flex-col" style={{ height: "40%" }}>
        <div className="flex flex-row mt-6 mb-4 ml-28 items-baseline">
          <h1 className="text-cyan-500 text-3xl font-abo-one">My Models</h1>
          <button
            onClick={() => handleNavigation("/models")}
            className="text-lg font-reem-kufi underline bg-opacity-0 text-gray-600 ml-12"
          >
            View All -&gt;
          </button>
        </div>
        <div className="flex justify-center items-center space-x-12">
          {loading ? (
            <div className="flex w-full py-6 items-center justify-center">
              <svg class="pl" width="240" height="240" viewBox="0 0 240 240">
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
              {models.map((item, index) => (
                <>
                  {item == model ? (
                    <div
                      key={model.id}
                      onClick={() => setModel(item)}
                      className="cursor-pointer flex flex-col items-center border-4 border-cyan-500 rounded-3xl py-3 px-2 font-reem-kufi"
                    >
                      <img
                        src="../../static/images/AiIcon.png"
                        className="w-2/5"
                      />
                      <p className="text-cyan-500 text-lg mb-2">{item.name}</p>
                      <p className="text-gray-600">{item.created_at}</p>
                    </div>
                  ) : (
                    <div
                      key={model.id}
                      onClick={() => setModel(item)}
                      className="cursor-pointer flex flex-col items-center border-4 border-cyan-300 rounded-3xl py-3 px-2 font-reem-kufi"
                    >
                      <img
                        src="../../static/images/AiIcon.png"
                        className="w-2/5"
                      />
                      <p className="text-cyan-500 text-lg mb-2">{item.name}</p>
                      <p className="text-gray-600">{item.created_at}</p>
                    </div>
                  )}
                </>
              ))}
            </>
          )}
        </div>
      </div>
      {/* Bottom Section */}
      <div className="flex items-center" style={{ height: "40%" }}>
        {/* Left Column */}
        <div className="flex w-2/3 justify-around px-12 items-center font-reem-kufi text-lg pt-4">
          <div className="text-center flex flex-col items-center">
            <EyeIcon fontSize="large" className="mb-2" />
            <button
              onClick={() => setIsPredictionPopupOpen(!isPredictionPopupOpen)}
              className="bg-cyan-300 hover:bg-cyan-200 py-4 px-8 rounded-2xl text-black transition duration-200"
            >
              Get Prediction
            </button>
          </div>
          <div className="text-center flex flex-col items-center">
            <UploadIcon fontSize="large" className="mb-2" />
            <button
              onClick={() => setIsStoreModelPopupOpen(!isStoreModelPopupOpen)}
              className="bg-yellow-300 hover:bg-yellow-200 py-4 px-8 rounded-2xl text-black transition duration-200"
            >
              Store Model
            </button>
          </div>
          <div className="text-center flex flex-col items-center">
            <DatabaseIcon fontSize="large" className="mb-2" />
            <button
              onClick={() => setIsInsertPopupOpen(!isInsertPopupOpen)}
              className="bg-mygreen hover:bg-lightgreen py-4 px-8 rounded-2xl text-black transition duration-200"
            >
              Insert Data
            </button>
          </div>
        </div>

        {/* Right Column */}
        <div className="flex w-1/3 flex-col justify-around items-center pt-8">
          <div className="border-4 border-cyan-600 w-4/5 p-4 rounded-3xl h-48">
            <div className="text-center mb-4 mt-2">
              <p className="font-reem-kufi text-2xl text-cyan-500">
                {model.name}
              </p>
            </div>
            <div className="text-center text-lg flex flex-row font-reem-kufi space-x-12">
              <div className="w-1/2 flex flex-col items-center justify-center">
                <AssignmentTurnedInIcon fontSize="large" className="mb-2" />
                <p>
                  <span className="text-brightgreen">{model.accuracy}</span>{" "}
                  <br /> Accuracy
                </p>
              </div>
              <div className="w-1/2 justify-center">
                <GridViewIcon fontSize="large" className="mb-2" />
                <p>
                  <span className="text-black">{model.size}</span>
                  <br /> Training Rows
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
      <Popup isOpen={isInsertPopupOpen} onClose={closeInsertPopup}>
        <InsertRowForm />
      </Popup>
      <Popup isOpen={isPredictionPopupOpen} onClose={closePredictionPopup}>
        <PredictionForm />
      </Popup>
      <Popup isOpen={isStoreModelPopupOpen} onClose={closeStoreModelPopup}>
        <StoreModel />
      </Popup>
    </div>
  );
};

export default AiPortal;
