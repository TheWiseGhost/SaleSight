import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import IconButton from "@mui/material/IconButton";
import AddIcon from "@mui/icons-material/Add";
import RemoveIcon from "@mui/icons-material/Remove";
import StoreModel from "./StoreModel";
import Popup from "./Popup";
import { useMediaQuery } from "react-responsive";

const ModelDisplay = () => {
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(false);
  const token = localStorage.getItem("token");
  const navigate = useNavigate();
  const isSmallScreen = useMediaQuery({ maxWidth: 768 });

  const [isOpenPopupOpen, setIsOpenPopupOpen] = useState(false);

  const openOpenPopup = () => setIsOpenPopupOpen(true);
  const closeOpenPopup = () => setIsOpenPopupOpen(false);

  const handleNavigation = (path) => {
    navigate(path);
  };

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        const response = await fetch("/api/models/", {
          method: "GET",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${token}`, // Include JWT in headers
          },
        });
        const res = await response.json();
        if (res.auth) {
          navigate("/auth");
        }
        if (res.models) {
          console.log(res.models);
          setData(res.models);
        }
        setLoading(false);
      } catch (error) {
        console.error("Error fetching data:", error);
      }
    };

    fetchData();
  }, [token]); // Ensure token is in the dependency array

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
    <div className="flex flex-col mt-8 mx-10 flex-grow">
      <div className="flex items-center w-full justify-between px-12">
        <h2 className="text-4xl font-abo-one w-1/3 text-cyan-600">
          All Models
        </h2>
        <div className="w-1/3 justify-center">
          <img
            src="../../static/images/SaleSightLogo.png"
            className="w-1/6 mx-auto"
          />
        </div>
        <div className="space-x-8 w-1/3 flex flex-row justify-end">
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
      </div>

      <div className="flex items-center pb-6 pt-10 w-full justify-between px-12">
        <div className="w-1/3 flex flex-row space-x-4 justify-start">
          <IconButton
            onClick={() => setIsOpenPopupOpen(true)}
            style={{
              backgroundColor: "#B2FF97",
              color: "black",
              width: "2.5rem",
              height: "2.5rem",
              marginTop: "auto",
            }}
          >
            <AddIcon fontSize="inherit" />
          </IconButton>
          <IconButton
            style={{
              backgroundColor: "#FF6C63",
              color: "black",
              width: "2.5rem",
              height: "2.5rem",
            }}
          >
            <RemoveIcon fontSize="inherit" />
          </IconButton>
        </div>
        <div className="w-1/3 justify-center">
          <h1 className="font-abo-one text-center text-black text-4xl">
            My Models
          </h1>
        </div>
        <div className="w-1/3"></div>
      </div>

      <div className="space-y-4 pt-8">
        <div className="px-16">
          <div className="grid grid-cols-4 gap-12 pb-16">
            {loading ? (
              <div className="flex col-span-4 justify-center items-center h-24">
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
              </div>
            ) : (
              <>
                {data.map((model, index) => (
                  <div
                    key={index}
                    className="w-full flex flex-col items-center border-4 border-cyan-300 rounded-3xl py-3 font-reem-kufi"
                  >
                    <img
                      src="../../static/images/AiIcon.png"
                      className="w-2/5"
                    />
                    <p className="text-cyan-500 text-lg mb-2">{model.name}</p>
                    <p className="text-gray-600">{model.created_at}</p>
                  </div>
                ))}
              </>
            )}
          </div>
        </div>
      </div>
      <Popup isOpen={isOpenPopupOpen} onClose={closeOpenPopup}>
        <StoreModel />
      </Popup>
    </div>
  );
};

export default ModelDisplay;
