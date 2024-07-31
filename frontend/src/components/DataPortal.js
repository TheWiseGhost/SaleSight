import React, { useEffect, useState } from "react";
import IconButton from "@mui/material/IconButton";
import AddIcon from "@mui/icons-material/Add";
import RemoveIcon from "@mui/icons-material/Remove";
import EditIcon from "@mui/icons-material/Edit";
import FileDownloadIcon from "@mui/icons-material/FileDownload";
import { useNavigate } from "react-router-dom";
import FeatureDisplay from "./FeatureDisplay";
import AddFeatureForm from "./AddFeatureForm";
import DeleteFeatureForm from "./DeleteFeatureForm";
import EditFeatureForm from "./EditFeatureForm";
import PredictionForm from "./PredictionForm";
import InsertFormRow from "./InsertRowForm";
import Popup from "./Popup";
import { useMediaQuery } from "react-responsive";

const DataPortal = () => {
  const [activeButton, setActiveButton] = useState("Offer");
  const navigate = useNavigate();
  const [loading, setLoading] = useState(false);
  const isSmallScreen = useMediaQuery({ maxWidth: 768 });

  const [features, setFeatures] = useState({
    responses: [],
    locations: [],
    lead_gens: [],
    offers: [],
    sale_products: [],
    funnel_stages: [],
    results: [],
  });

  const [isInsertPopupOpen, setIsInsertPopupOpen] = useState(false);
  const [isPredictionPopupOpen, setIsPredictionPopupOpen] = useState(false);
  const [isAddFeaturePopupOpen, setIsAddFeaturePopupOpen] = useState(false);
  const [isDeleteFeaturePopupOpen, setIsDeleteFeaturePopupOpen] =
    useState(false);
  const [isEditFeaturePopupOpen, setIsEditFeaturePopupOpen] = useState(false);

  const openInsertPopup = () => setIsInsertPopupOpen(true);
  const closeInsertPopup = () => setIsInsertPopupOpen(false);

  const openPredictionPopup = () => setIsPredictionPopupOpen(true);
  const closePredictionPopup = () => setIsPredictionPopupOpen(false);

  const openAddFeaturePopup = () => setIsAddFeaturePopupOpen(true);
  const closeAddFeaturePopup = () => setIsAddFeaturePopupOpen(false);

  const openDeleteFeaturePopup = () => setIsDeleteFeaturePopupOpen(true);
  const closeDeleteFeaturePopup = () => setIsDeleteFeaturePopupOpen(false);

  const openEditFeaturePopup = () => setIsEditFeaturePopupOpen(true);
  const closeEditFeaturePopup = () => setIsEditFeaturePopupOpen(false);

  const translator = {
    Response: "Responses",
    Location: "Locations",
    Lead: "Lead_Gens",
    Offer: "Offers",
    Product: "Sale_Products",
    "Funnel Stage": "Funnel_Stages",
    Result: "Results",
  };

  useEffect(() => {
    const token = localStorage.getItem("token"); // Retrieve the token from localStorage

    const fetchFeatures = async () => {
      try {
        setLoading(true);
        const response = await fetch("/api/get_feature_values/", {
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
        console.log(data);
        setFeatures(data || []);
        setLoading(false);
      } catch (error) {
        console.error("Error fetching recent data:", error);
      }
    };

    fetchFeatures();
  }, []);

  const renderFeatures = (options) => {
    try {
      console.log(options);
      if (options != "[]") {
        const parsedOptions = JSON.parse(options);
        return parsedOptions.map((option) => (
          <div className="font-reem-kufi" key={option.main} value={option.main}>
            {option.main}
          </div>
        ));
      } else {
        return (
          <div className="font-reem-kufi">Add your first {activeButton}</div>
        );
      }
    } catch (error) {
      console.error("Error parsing options:", error);
      return null;
    }
  };

  const handleButtonClick = (buttonName) => {
    setActiveButton(buttonName);
  };

  const handleNavigation = (path) => {
    navigate(path);
  };

  const handleExport = async () => {
    const token = localStorage.getItem("token");
    try {
      const response = await fetch("/api/export_csv/", {
        method: "GET",
        headers: {
          "Content-Type": "text/csv",
          Authorization: `Bearer ${token}`,
        },
      });

      if (!response.ok) {
        throw new Error("Network response was not ok");
      }

      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.style.display = "none";
      a.href = url;
      a.download = "salesight_data.csv";
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
    } catch (error) {
      console.error("There was an error exporting the CSV:", error);
    }
  };

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
    <div className="flex h-screen pt-6">
      {/* Left Column */}
      <div className="w-2/5">
        <h1 className="text-4xl text-brightgreen pt-4 font-abo-one ml-16">
          Data Portal
        </h1>
        <div className="pt-6 ml-24">
          <h2 className="text-2xl font-reem-kufi underline">My Data</h2>
          <div className="flex items-center mt-4 space-x-12">
            <IconButton
              onClick={() => setIsInsertPopupOpen(!isInsertPopupOpen)}
              style={{
                backgroundColor: "#B2FF97",
                color: "black",
                width: "4rem",
                height: "4rem",
              }}
            >
              <AddIcon fontSize="inherit" />
            </IconButton>
            <IconButton
              onClick={() => handleNavigation("/data")}
              style={{
                backgroundColor: "#FF6C63",
                color: "black",
                width: "4rem",
                height: "4rem",
              }}
            >
              <RemoveIcon fontSize="inherit" />
            </IconButton>
            <IconButton
              onClick={() => handleExport()}
              style={{
                backgroundColor: "#D9D9D9",
                color: "black",
                width: "4rem",
                height: "4rem",
              }}
            >
              <FileDownloadIcon fontSize="inherit" />
            </IconButton>
          </div>
          <h2 className="text-2xl mt-12 font-reem-kufi underline">
            My Features
          </h2>
          <div className="flex items-center mt-4 space-x-12">
            <IconButton
              onClick={() => setIsAddFeaturePopupOpen(!isAddFeaturePopupOpen)}
              style={{
                backgroundColor: "#B2FF97",
                color: "black",
                width: "4rem",
                height: "4rem",
              }}
            >
              <AddIcon fontSize="inherit" />
            </IconButton>
            <IconButton
              onClick={() =>
                setIsDeleteFeaturePopupOpen(!isDeleteFeaturePopupOpen)
              }
              style={{
                backgroundColor: "#FF6C63",
                color: "black",
                width: "4rem",
                height: "4rem",
              }}
            >
              <RemoveIcon fontSize="inherit" />
            </IconButton>
            <IconButton
              onClick={() => setIsEditFeaturePopupOpen(!isEditFeaturePopupOpen)}
              style={{
                backgroundColor: "#D9D9D9",
                color: "black",
                width: "4rem",
                height: "4rem",
              }}
            >
              <EditIcon fontSize="inherit" />
            </IconButton>
          </div>
          <button
            onClick={() => handleNavigation("/data")}
            className="w-4/5 mt-16 py-4 font-abo-one text-lg bg-mygreen hover:bg-lightgreen transition duration-200 rounded-2xl"
          >
            View all Data
          </button>
        </div>
      </div>

      {/* Middle Column */}
      <div className="w-1/5 flex flex-col items-center">
        <div className="mt-4">
          <img
            src="../../static/images/SaleSightLogo.png"
            alt="Logo"
            className="w-1/4 mx-auto"
          />
        </div>
        <div className="flex flex-col h-full items-center mb-24 w-3/4 mx-auto mt-4 p-4 border-4 border-gray-200 rounded-xl">
          <h2 className="text-2xl font-abo-one text-center">{activeButton}</h2>
          <div className="mt-6 space-y-4 text-lg text-center font-reem-kufi">
            {loading ? (
              <div className="flex w-full pt-12 items-center justify-center">
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
                {renderFeatures(
                  JSON.stringify(features[translator[activeButton]] || [])
                )}
              </>
            )}
          </div>
          <IconButton
            onClick={() => setIsAddFeaturePopupOpen(!isAddFeaturePopupOpen)}
            style={{
              backgroundColor: "#B2FF97",
              color: "black",
              width: "3rem",
              height: "3rem",
              marginTop: "auto",
            }}
          >
            <AddIcon fontSize="inherit" />
          </IconButton>
        </div>
      </div>

      {/* Right Column */}
      <div className="w-2/5 flex flex-col">
        <div className="flex space-x-8 pt-4 justify-end mr-16">
          <button
            onClick={() => handleNavigation("/aiportal")}
            className="py-3 font-abo-one px-6 bg-cyan-300 hover:bg-cyan-200 transition duration-200 rounded-xl"
          >
            Ai Portal
          </button>
          <button
            onClick={() => handleNavigation("/dashboard")}
            className="py-3 font-abo-one px-6 bg-yellow-300 hover:bg-yellow-200 transition duration-200 rounded-xl"
          >
            Home
          </button>
        </div>
        <h2 className="text-2xl font-reem-kufi underline mt-8 mr-24 ml-12 text-center">
          Insights
        </h2>
        <div className="flex-grow overflow-x-hidden overflow-y-hidden h-full mb-24 mt-4 pl-2 pt-2 border-gray-300 border-4 rounded-lg mr-24 ml-12">
          <FeatureDisplay />
        </div>
      </div>

      {/* Bottom Row */}
      <div className="absolute font-reem-kufi bottom-0 left-0 right-0 flex justify-center space-x-1 bg-gray-500 mx-20 my-4 px-1 py-1">
        {[
          "Offer",
          "Product",
          "Location",
          "Lead",
          "Funnel Stage",
          "Response",
          "Result",
        ].map((buttonName) => (
          <button
            key={buttonName}
            onClick={() => handleButtonClick(buttonName)}
            className={`py-2 px-4 w-full ${
              activeButton === buttonName ? "bg-lightgreen" : "bg-white"
            }`}
          >
            {buttonName}
          </button>
        ))}
      </div>
      <Popup isOpen={isInsertPopupOpen} onClose={closeInsertPopup}>
        <InsertFormRow />
      </Popup>
      <Popup isOpen={isPredictionPopupOpen} onClose={closePredictionPopup}>
        <PredictionForm />
      </Popup>
      <Popup isOpen={isAddFeaturePopupOpen} onClose={closeAddFeaturePopup}>
        <AddFeatureForm />
      </Popup>
      <Popup
        isOpen={isDeleteFeaturePopupOpen}
        onClose={closeDeleteFeaturePopup}
      >
        <DeleteFeatureForm />
      </Popup>
      <Popup isOpen={isEditFeaturePopupOpen} onClose={closeEditFeaturePopup}>
        <EditFeatureForm />
      </Popup>
    </div>
  );
};

export default DataPortal;
